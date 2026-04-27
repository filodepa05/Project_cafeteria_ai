"""
demo.py – Smart Tray · Pitch Demo

Usage:
    py demo.py
    py demo.py --checkpoint checkpoints/epoch_030_loss_0.2145.pt
    py demo.py --no-share
    py demo.py --port 8080
"""

import argparse
import base64
import io
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from src.config import load_config
from src.dataset import CATEGORIES
from src.models.tray_model import TrayModel
from src.nlp_summary import generate_summary
from src.nutrition import estimate_nutrition
from src.utils.io import resolve_device

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--config",     type=str, default="configs/base.yaml")
parser.add_argument("--no-share",   action="store_true")
parser.add_argument("--port",       type=int, default=7860)
args = parser.parse_args()

CFG = load_config(args.config)
DEV = resolve_device(CFG.inference.device)

print("\n" + "=" * 55)
print("  SmartTray · Starting up")
print("=" * 55)
print(f"  Device  : {DEV}")

# ── YOLO ─────────────────────────────────────────────────────────
YOLO = None
try:
    from src.models.yolo_detector import YOLOFoodDetector
    w = Path(CFG.yolo.weights_path)
    if w.exists():
        YOLO = YOLOFoodDetector(str(w),
            conf_threshold=CFG.inference.confidence_threshold,
            iou_threshold=CFG.inference.nms_iou_threshold)
        print(f"  YOLO    : loaded  ({w.name})")
    else:
        print(f"  YOLO    : weights not found at {w}")
except Exception as e:
    print(f"  YOLO    : {e}")

# ── ResNet ────────────────────────────────────────────────────────
RESNET = TrayModel(CFG.model).to(DEV)
RESNET.eval()
resnet_loaded = False

def _best_ckpt(d):
    pts = sorted(Path(d).glob("epoch_*_loss_*.pt"))
    if not pts: return None
    return min(pts, key=lambda p: float(p.stem.split("loss_")[1]) if "loss_" in p.stem else 99)

ckpt = Path(args.checkpoint) if args.checkpoint else _best_ckpt(CFG.checkpoint.save_dir)
if ckpt and Path(ckpt).exists():
    try:
        sd = torch.load(ckpt, map_location=DEV, weights_only=False)
        RESNET.load_state_dict(sd["model_state_dict"])
        resnet_loaded = True
        print(f"  ResNet  : loaded  ({Path(ckpt).name})")
    except Exception as e:
        print(f"  ResNet  : {e}")
else:
    print("  ResNet  : no checkpoint — random weights")

print("=" * 55 + "\n")

TF = T.Compose([
    T.Resize((CFG.data.image_size, CFG.data.image_size)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

DOT_COLORS = ["#2d6a4f","#40916c","#52b788","#b5813d","#9b3a2a",
              "#1b4332","#74c69d","#6b4f2a","#40916c","#2d6a4f"]

DEFAULT_GRAMS = {
    "pasta":250,"rice":220,"pizza":200,"bread":80,"fries":120,
    "couscous":200,"potatoes":180,"wrap_sandwich":200,
    "grilled_chicken":170,"fried_chicken":180,"chicken_stew":250,
    "turkey":160,"grilled_beef":180,"beef_stew":250,"meatballs":200,
    "grilled_pork":170,"pork_ribs":220,"salmon":160,"hake":160,
    "tuna":150,"cod":160,"grilled_fish":150,"fried_fish":160,
    "eggs":100,"lentils":250,"chickpeas":220,"salad":130,
    "soup_cream":280,"grilled_vegetables":150,"sauteed_vegetables":150,
    "broccoli":150,"stuffed_peppers":200,"poke_bowl":350,
    "lasagne":300,"fresh_fruit":150,"fruit_salad":180,"yogurt":125,
    "cake_pastry":100,"ice_cream_sorbet":120,"juice_drink":250,
    "rotisserie_chicken":250,"fried_potatoes":150,"baked_potatoes":180,
}

def _font(size=14):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

FONT = _font(15)

@torch.no_grad()
def run_resnet(img, thr):
    t = TF(img).unsqueeze(0).to(DEV)
    o = RESNET(t)
    probs = torch.sigmoid(o["logits"][0]).cpu()
    g = max(30., min(float(o["grams"][0,0].cpu()), 400.))
    return [{"label":CATEGORIES[i],"grams":DEFAULT_GRAMS.get(CATEGORIES[i],round(g,1)),"confidence":round(p.item(),3)}
            for i,p in enumerate(probs) if p.item()>=thr]

def run_yolo(img, thr):
    out = []
    for d in YOLO.detect(img):
        if d.confidence < thr: continue
        x1,y1,x2,y2 = d.bbox
        area = max((x2-x1)*(y2-y1),1)
        g = DEFAULT_GRAMS.get(d.label, round(max(30.,min(area/(img.width*img.height)*800.,400.)),1))
        out.append({"label":d.label,"grams":g,"confidence":round(d.confidence,3),"bbox":(x1,y1,x2,y2)})
    return out

def annotate(img, items):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, item in enumerate(items):
        c = DOT_COLORS[i % len(DOT_COLORS)]
        name = item["label"].replace("_"," ").title()
        if "bbox" in item:
            x1,y1,x2,y2 = [int(v) for v in item["bbox"]]
            draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
            txt = f"{name} ~{item['grams']:.0f}g"
            tw = len(txt)*8
            draw.rectangle([x1,y1-26,x1+tw+10,y1], fill=c)
            draw.text((x1+5,y1-22), txt, fill="white", font=FONT)
        else:
            yp = 12+i*30
            txt = f"{name} ~{item['grams']:.0f}g"
            tw = len(txt)*8
            draw.rectangle([8,yp-2,20+tw,yp+24], fill=c)
            draw.text((13,yp), txt, fill="white", font=FONT)
    return out

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def health_score(t, n):
    if not n: return 0
    cal,prot,fat,carb = t["calories"],t["protein_g"],t["fat_g"],t["carbs_g"]
    s = 0
    if 600<=cal<=900: s+=30
    elif 300<=cal<=1200: s+=15
    if prot>=35: s+=25
    elif prot>=25: s+=20
    elif prot>=15: s+=12
    elif prot>=8: s+=5
    fp = fat*9/cal*100 if cal>0 else 0
    if 25<=fp<=40: s+=25
    elif 15<=fp<=55: s+=12
    cp = carb*4/cal*100 if cal>0 else 0
    if 40<=cp<=55: s+=20
    elif 20<=cp<=75: s+=10
    return max(0,min(100,s))

def score_meta(s):
    if s>=80: return "Excellent","#2d6a4f"
    if s>=60: return "Good","#40916c"
    if s>=40: return "Moderate","#b5813d"
    return "Needs work","#9b3a2a"

def analyse(image, threshold, use_yolo):
    if image is None:
        return EMPTY_HTML

    img = Image.fromarray(image).convert("RGB")
    raw = run_yolo(img, threshold) if (use_yolo and YOLO) else run_resnet(img, threshold)

    if not raw:
        return """<div style="display:flex;align-items:center;justify-content:center;min-height:300px;
                              color:#a89880;font-size:15px;font-family:'Georgia',serif">
                    Nothing detected — try lowering the confidence threshold
                  </div>"""

    items_out, totals = [], {"calories":0.,"protein_g":0.,"carbs_g":0.,"fat_g":0.}
    for item in raw:
        cid = CATEGORIES.index(item["label"]) if item["label"] in CATEGORIES else -1
        n = estimate_nutrition(cid, item["grams"])
        entry = {"food":item["label"],"grams":item["grams"],
                 "calories":n.calories,"protein_g":n.protein_g,
                 "carbs_g":n.carbs_g,"fat_g":n.fat_g}
        if "bbox" in item: entry["bbox"] = item["bbox"]
        items_out.append(entry)
        for k in totals: totals[k] += entry[k]

    totals = {k:round(v,1) for k,v in totals.items()}
    ann = annotate(img, raw)
    ann_b64 = img_to_b64(ann)
    is_yolo = any("bbox" in x for x in items_out)
    return build_html(items_out, totals, ann_b64, is_yolo)

def build_html(items, totals, ann_b64, is_yolo):
    s = health_score(totals, len(items))
    label, color = score_meta(s)
    summary = generate_summary({"items":items,"totals":totals})
    max_cal = max((x["calories"] for x in items), default=1)
    mode_txt = "YOLO" if is_yolo else "ResNet"

    rows = ""
    for i, item in enumerate(items):
        c = DOT_COLORS[i % len(DOT_COLORS)]
        name = item["food"].replace("_"," ").title()
        bw = int(item["calories"]/max_cal*100)
        rows += f"""
        <tr>
          <td style="padding:12px 18px;border-bottom:1px solid #e8ddd0">
            <div style="display:flex;align-items:center;gap:10px">
              <div style="width:9px;height:9px;border-radius:50%;background:{c};flex-shrink:0"></div>
              <span style="font-weight:600;color:#2c2416;font-size:14px;font-family:'Georgia',serif">{name}</span>
            </div>
            <div style="color:#a89880;font-size:11px;margin-top:2px;padding-left:19px">{item['grams']}g</div>
          </td>
          <td style="padding:12px 18px;text-align:right;border-bottom:1px solid #e8ddd0;vertical-align:middle">
            <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
              <div style="width:56px;height:3px;background:#e8ddd0;border-radius:2px">
                <div style="width:{bw}%;height:100%;background:{c};border-radius:2px"></div>
              </div>
              <span style="color:#2d6a4f;font-weight:800;font-size:15px;min-width:44px;text-align:right;font-family:system-ui">{item['calories']}</span>
            </div>
          </td>
          <td style="padding:12px 18px;text-align:center;color:#40916c;font-size:13px;font-weight:600;border-bottom:1px solid #e8ddd0;font-family:system-ui">{item['protein_g']}g</td>
          <td style="padding:12px 18px;text-align:center;color:#b5813d;font-size:13px;font-weight:600;border-bottom:1px solid #e8ddd0;font-family:system-ui">{item['carbs_g']}g</td>
          <td style="padding:12px 18px;text-align:center;color:#9b3a2a;font-size:13px;font-weight:600;border-bottom:1px solid #e8ddd0;font-family:system-ui">{item['fat_g']}g</td>
        </tr>"""

    return f"""
<div style="font-family:'Georgia',serif;color:#2c2416;background:#f5f0e8;border-radius:18px;overflow:hidden;border:1px solid #ddd3c0">

  <div style="position:relative">
    <img src="{ann_b64}" style="width:100%;display:block;max-height:320px;object-fit:cover">
    <div style="position:absolute;top:12px;left:12px;background:rgba(245,240,232,0.93);
                border-radius:99px;padding:4px 13px;font-size:10px;letter-spacing:1.5px;
                font-weight:700;color:#2d6a4f;font-family:system-ui">
      {mode_txt.upper()}
    </div>
  </div>

  <div style="padding:22px">

    <div style="display:grid;grid-template-columns:auto 1fr;gap:14px;margin-bottom:18px">
      <div style="background:white;border-radius:14px;padding:18px 20px;text-align:center;border:1px solid #e0d5c5;min-width:112px">
        <div style="font-size:46px;font-weight:800;color:{color};line-height:1;font-family:system-ui">{s}</div>
        <div style="font-size:9px;letter-spacing:2px;color:#a89880;margin-top:4px;font-family:system-ui">HEALTH SCORE</div>
        <div style="font-size:13px;font-weight:700;color:{color};margin-top:6px">{label}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="background:white;border:1px solid #e0d5c5;border-radius:11px;padding:12px;text-align:center">
          <div style="font-size:21px;font-weight:800;color:#2d6a4f;font-family:system-ui">{totals['calories']}</div>
          <div style="font-size:9px;color:#a89880;letter-spacing:1.5px;margin-top:2px;font-family:system-ui">KCAL</div>
        </div>
        <div style="background:white;border:1px solid #e0d5c5;border-radius:11px;padding:12px;text-align:center">
          <div style="font-size:21px;font-weight:800;color:#40916c;font-family:system-ui">{totals['protein_g']}g</div>
          <div style="font-size:9px;color:#a89880;letter-spacing:1.5px;margin-top:2px;font-family:system-ui">PROTEIN</div>
        </div>
        <div style="background:white;border:1px solid #e0d5c5;border-radius:11px;padding:12px;text-align:center">
          <div style="font-size:21px;font-weight:800;color:#b5813d;font-family:system-ui">{totals['carbs_g']}g</div>
          <div style="font-size:9px;color:#a89880;letter-spacing:1.5px;margin-top:2px;font-family:system-ui">CARBS</div>
        </div>
        <div style="background:white;border:1px solid #e0d5c5;border-radius:11px;padding:12px;text-align:center">
          <div style="font-size:21px;font-weight:800;color:#9b3a2a;font-family:system-ui">{totals['fat_g']}g</div>
          <div style="font-size:9px;color:#a89880;letter-spacing:1.5px;margin-top:2px;font-family:system-ui">FAT</div>
        </div>
      </div>
    </div>

    <div style="background:white;border:1px solid #d4c9b5;border-left:4px solid #40916c;
                border-radius:12px;padding:14px 18px;margin-bottom:18px">
      <div style="font-size:10px;color:#40916c;letter-spacing:2px;font-weight:700;margin-bottom:7px;font-family:system-ui">AI NUTRITIONAL SUMMARY</div>
      <div style="color:#4a3f2f;font-size:14px;line-height:1.75">{summary}</div>
    </div>

    <div style="background:white;border-radius:14px;overflow:hidden;border:1px solid #e0d5c5">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:#f0ebe0;border-bottom:2px solid #e0d5c5">
            <th style="padding:11px 18px;text-align:left;font-size:10px;letter-spacing:1.5px;color:#a89880;font-weight:700;font-family:system-ui">ITEM</th>
            <th style="padding:11px 18px;text-align:right;font-size:10px;letter-spacing:1.5px;color:#2d6a4f;font-weight:700;font-family:system-ui">KCAL</th>
            <th style="padding:11px 18px;text-align:center;font-size:10px;letter-spacing:1.5px;color:#40916c;font-weight:700;font-family:system-ui">PROTEIN</th>
            <th style="padding:11px 18px;text-align:center;font-size:10px;letter-spacing:1.5px;color:#b5813d;font-weight:700;font-family:system-ui">CARBS</th>
            <th style="padding:11px 18px;text-align:center;font-size:10px;letter-spacing:1.5px;color:#9b3a2a;font-weight:700;font-family:system-ui">FAT</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
        <tfoot>
          <tr style="background:#f0ebe0;border-top:2px solid #e0d5c5">
            <td style="padding:13px 18px;font-weight:700;color:#2c2416;font-size:14px">Total</td>
            <td style="padding:13px 18px;text-align:right;font-weight:800;color:#2d6a4f;font-size:17px;font-family:system-ui">{totals['calories']}</td>
            <td style="padding:13px 18px;text-align:center;font-weight:700;color:#40916c;font-family:system-ui">{totals['protein_g']}g</td>
            <td style="padding:13px 18px;text-align:center;font-weight:700;color:#b5813d;font-family:system-ui">{totals['carbs_g']}g</td>
            <td style="padding:13px 18px;text-align:center;font-weight:700;color:#9b3a2a;font-family:system-ui">{totals['fat_g']}g</td>
          </tr>
        </tfoot>
      </table>
    </div>

  </div>
</div>"""

EMPTY_HTML = """
<div style="display:flex;align-items:center;justify-content:center;min-height:420px;
            flex-direction:column;gap:14px;background:#f5f0e8;border-radius:18px;
            border:2px dashed #d4c9b5">
  <div style="font-size:48px;opacity:0.2">🍽️</div>
  <div style="color:#a89880;font-size:15px;font-family:'Georgia',serif">Upload a tray photo and click Analyse</div>
</div>"""

# ── CSS ───────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&display=swap');

body, .gradio-container, #root, .main, .wrap, .app {
    background: #ede8df !important;
    font-family: 'Lora', Georgia, serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
footer, .built-with, .svelte-1ax1toq { display: none !important; }

.gr-box, .gr-form, .gr-panel, .block, .gap, .gr-padded, div.gr-block {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}

[data-testid="image"], .gr-image, .upload-container, .svelte-xpkpmc {
    background: #f5f0e8 !important;
    border: 2px dashed #c4b8a8 !important;
    border-radius: 16px !important;
    min-height: 320px !important;
}
[data-testid="image"]:hover { border-color: #40916c !important; }

button[variant="primary"], .gr-button-primary {
    background: #2d6a4f !important; color: #f5f0e8 !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 15px !important;
    padding: 14px 28px !important; width: 100% !important;
    font-family: 'Lora', Georgia, serif !important;
    transition: all 0.18s !important;
    box-shadow: 0 4px 20px rgba(45,106,79,0.25) !important;
}
button[variant="primary"]:hover {
    background: #1b4332 !important; transform: translateY(-1px) !important;
}

details, .gr-accordion {
    background: #f5f0e8 !important; border: 1px solid #d4c9b5 !important;
    border-radius: 12px !important;
}
details summary { padding: 10px 16px !important; color: #a89880 !important; font-size: 13px !important; cursor: pointer !important; }

input[type=range] { accent-color: #40916c !important; }
input[type=checkbox] { accent-color: #40916c !important; }
label span { color: #a89880 !important; font-size: 12px !important; }

.gr-html, [data-testid="html"] { background: transparent !important; border: none !important; padding: 0 !important; }

.gr-examples { background: transparent !important; border: none !important; }
.gr-examples img { border-radius: 10px !important; border: 1px solid #d4c9b5 !important; }
"""

yolo_ok = YOLO is not None

with gr.Blocks(css=css, title="SmartTray") as app:

    gr.HTML(f"""
    <div style="background:#ede8df;padding:48px 48px 32px;font-family:'Lora',Georgia,serif;border-bottom:1px solid #d4c9b5;margin-bottom:32px">
      <div style="max-width:1200px;margin:0 auto;text-align:center">
        <div style="font-size:clamp(42px,5vw,62px);font-weight:700;color:#1b4332;letter-spacing:-1px;line-height:1.05;margin-bottom:12px">
          Smart<span style="color:#40916c">Tray</span>
        </div>
        <div style="color:#7a6a55;font-size:17px;max-width:480px;margin:0 auto 20px;line-height:1.7;font-style:italic">
          AI system that scans cafeteria trays and estimates calories and nutrition in real time
        </div>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
          <span style="background:{'#d8f3dc' if yolo_ok else '#ede8df'};color:{'#1b4332' if yolo_ok else '#a89880'};border:1px solid {'#95d5b2' if yolo_ok else '#d4c9b5'};font-size:11px;padding:4px 14px;border-radius:99px;font-weight:600;letter-spacing:1px;font-family:system-ui">
            YOLO {'active' if yolo_ok else 'unavailable'}
          </span>
          <span style="background:{'#d8f3dc' if resnet_loaded else '#fdf3e3'};color:{'#1b4332' if resnet_loaded else '#b5813d'};border:1px solid {'#95d5b2' if resnet_loaded else '#e8c97a'};font-size:11px;padding:4px 14px;border-radius:99px;font-weight:600;letter-spacing:1px;font-family:system-ui">
            ResNet {'loaded' if resnet_loaded else 'random weights'}
          </span>
        </div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="", type="numpy", height=320, sources=["upload","clipboard"])
            with gr.Accordion("Options", open=False):
                threshold = gr.Slider(0.05, 0.9, value=CFG.inference.confidence_threshold,
                                      step=0.05, label="Confidence threshold")
                use_yolo = gr.Checkbox(value=yolo_ok,
                    label=f"Use YOLO detector ({'active' if yolo_ok else 'unavailable'})",
                    interactive=yolo_ok)
            run_btn = gr.Button("Analyse Tray", variant="primary")

        with gr.Column(scale=1):
            result_html = gr.HTML(value=EMPTY_HTML)

    example_dir = Path("examples")
    examples = [str(p) for p in sorted(example_dir.glob("tray_*.*"))[:6]] if example_dir.exists() else []
    if examples:
        gr.Examples(examples=examples, inputs=image_input, label="Example trays")

    gr.HTML("""
    <div style="text-align:center;padding:28px;color:#c4b8a8;font-size:12px;letter-spacing:0.5px;
                font-family:system-ui;border-top:1px solid #d4c9b5;margin-top:32px;background:#ede8df">
      IE University &middot; AI Project &middot; Filo &middot; Nicolas &middot; Yago &middot; Dimash &middot; JP &middot; Santi
    </div>""")

    run_btn.click(fn=analyse, inputs=[image_input, threshold, use_yolo], outputs=[result_html])

# NEW — remove css from launch()
if __name__ == "__main__":
    app.launch(share=not args.no_share, server_port=args.port)