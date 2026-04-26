"""
demo.py – Smart Tray · Pitch Demo
py demo.py  →  http://127.0.0.1:7860
"""

from pathlib import Path
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from src.config import load_config
from src.dataset import CATEGORIES
from src.nutrition import estimate_nutrition
from src.nlp_summary import generate_summary
from src.models.tray_model import TrayModel
from src.utils.io import resolve_device
import torch
from torchvision import transforms as T

try:
    from src.models.yolo_detector import YOLOFoodDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CFG  = load_config("configs/base.yaml")
DEV  = resolve_device(CFG.inference.device)
RESNET = TrayModel(CFG.model).to(DEV)
RESNET.eval()

YOLO = None
if YOLO_AVAILABLE:
    w = Path(CFG.yolo.weights_path)
    if w.exists():
        try:
            YOLO = YOLOFoodDetector(str(w),
                conf_threshold=CFG.inference.confidence_threshold,
                iou_threshold=CFG.inference.nms_iou_threshold)
            print(f"YOLO loaded ✓")
        except Exception as e:
            print(f"YOLO failed: {e}")

ckpt_dir = Path(CFG.checkpoint.save_dir)
ckpts = sorted(ckpt_dir.glob("epoch_*_loss_*.pt")) if ckpt_dir.exists() else []
if ckpts:
    def _l(p):
        try: return float(p.stem.split("loss_")[1])
        except: return float("inf")
    best = min(ckpts, key=_l)
    try:
        ck = torch.load(best, map_location=DEV, weights_only=False)
        RESNET.load_state_dict(ck["model_state_dict"])
        print(f"ResNet loaded ✓  ({best.name})")
    except Exception as e:
        print(f"ResNet ckpt failed: {e}")

TF = T.Compose([
    T.Resize((CFG.data.image_size, CFG.data.image_size)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

COLORS = ["#6C63FF","#00D4AA","#FF6B6B","#FFB347","#4FC3F7",
          "#F06292","#AED581","#FF8A65","#BA68C8","#4DB6AC"]

DEFAULT_GRAMS = {
    "pasta": 250, "rice": 220, "pizza": 200, "bread": 80, "fries": 120,
    "couscous": 200, "potatoes": 180, "wrap_sandwich": 200,
    "grilled_chicken": 170, "fried_chicken": 180, "chicken_stew": 250,
    "turkey": 160, "grilled_beef": 180, "beef_stew": 250, "meatballs": 200,
    "grilled_pork": 170, "pork_ribs": 220, "salmon": 160, "hake": 160,
    "tuna": 150, "cod": 160, "grilled_fish": 150, "fried_fish": 160,
    "eggs": 100, "lentils": 250, "chickpeas": 220, "salad": 130,
    "soup_cream": 280, "grilled_vegetables": 150, "sauteed_vegetables": 150,
    "broccoli": 150, "stuffed_peppers": 200, "poke_bowl": 350,
    "lasagne": 300, "fresh_fruit": 150, "fruit_salad": 180, "yogurt": 125,
    "cake_pastry": 100, "ice_cream_sorbet": 120, "juice_drink": 250,
    "rotisserie_chicken": 250, "fried_potatoes": 150, "baked_potatoes": 180,
}

@torch.no_grad()
def run_resnet(img, thr):
    t = TF(img).unsqueeze(0).to(DEV)
    o = RESNET(t)
    probs = torch.sigmoid(o["logits"][0]).cpu()
    g = max(30., min(float(o["grams"][0,0].cpu()), 400.))
    return [{"label": CATEGORIES[i], "grams": round(g,1), "confidence": round(p.item(),3)}
            for i,p in enumerate(probs) if p.item() >= thr]

def run_yolo(img, threshold):
    out = []
    for d in YOLO.detect(img):
        if d.confidence >= threshold:
            x1,y1,x2,y2 = d.bbox
            area = (x2-x1)*(y2-y1)
            fallback = round(max(30., min(area/(img.width*img.height)*800., 400.)), 1)
            grams = DEFAULT_GRAMS.get(d.label, fallback)
            out.append({"label":d.label,"grams":grams,"confidence":round(d.confidence,3),"bbox":(x1,y1,x2,y2)})
    return out

def annotate(img, items):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try: font = ImageFont.truetype("arial.ttf", 15)
    except: font = ImageFont.load_default()
    for i, item in enumerate(items):
        c = COLORS[i % len(COLORS)]
        label = item["label"].replace("_"," ").title()
        g = item["grams"]
        if "bbox" in item:
            x1,y1,x2,y2 = item["bbox"]
            draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
            txt = f"{label}  ~{g:.0f}g"
            tw = len(txt)*8
            draw.rectangle([x1, y1-22, x1+tw+6, y1], fill=c)
            draw.text((x1+3, y1-20), txt, fill="white", font=font)
        else:
            yp = 10+i*26
            txt = f"{label}  ~{g:.0f}g"
            tw = len(txt)*8
            draw.rectangle([8,yp-2,16+tw,yp+20], fill=c)
            draw.text((12,yp), txt, fill="white", font=font)
    return out

def health_score(totals, n_items):
    if n_items == 0:
        return 0

    cal   = totals["calories"]
    prot  = totals["protein_g"]
    fat   = totals["fat_g"]
    carb  = totals["carbs_g"]

    score = 0

    # 1. Calories (target: 600-900 kcal for a main meal)
    if 600 <= cal <= 900:
        score += 30
    elif 450 <= cal < 600 or 900 < cal <= 1050:
        score += 20
    elif 300 <= cal < 450 or 1050 < cal <= 1200:
        score += 10
    else:
        score += 0  # too low or too high

    # 2. Protein (target: 25g+ for satiety and muscle)
    if prot >= 35:
        score += 25
    elif prot >= 25:
        score += 20
    elif prot >= 15:
        score += 12
    elif prot >= 8:
        score += 5
    else:
        score += 0

    # 3. Fat ratio (target: 25-40% of calories)
    fat_pct = (fat * 9 / cal * 100) if cal > 0 else 0
    if 25 <= fat_pct <= 40:
        score += 25
    elif 20 <= fat_pct < 25 or 40 < fat_pct <= 45:
        score += 15
    elif 15 <= fat_pct < 20 or 45 < fat_pct <= 55:
        score += 8
    else:
        score += 0  # very low fat or very high fat

    # 4. Carb ratio (target: 40-55% of calories)
    carb_pct = (carb * 4 / cal * 100) if cal > 0 else 0
    if 40 <= carb_pct <= 55:
        score += 20
    elif 30 <= carb_pct < 40 or 55 < carb_pct <= 65:
        score += 12
    elif 20 <= carb_pct < 30 or 65 < carb_pct <= 75:
        score += 6
    else:
        score += 0

    return max(0, min(100, score))

def score_label(s):
    if s >= 80: return "EXCELLENT", "#00D4AA"
    if s >= 60: return "GOOD", "#6C63FF"
    if s >= 40: return "MODERATE", "#FFB347"
    return "REVIEW DIET", "#FF6B6B"

def build_output(items_out, totals, annotated_img):
    s = health_score(totals, len(items_out))
    label, color = score_label(s)

    rows = ""
    for i, item in enumerate(items_out):
        bg = "rgba(255,255,255,0.04)" if i%2==0 else "transparent"
        name = item["food"].replace("_"," ").title()
        bar_w = min(100, int(item["calories"] / max(totals["calories"],1) * 100 * len(items_out)))
        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:10px 14px;font-weight:600;color:#f0f0f0;font-size:14px">{name}</td>
          <td style="padding:10px 14px;text-align:center;color:#aaa;font-size:13px">{item['grams']}g</td>
          <td style="padding:10px 14px;text-align:right;font-size:13px">
            <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
              <div style="width:60px;height:4px;background:rgba(255,255,255,0.1);border-radius:2px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;background:{COLORS[i%len(COLORS)]};border-radius:2px"></div>
              </div>
              <span style="color:#ff8a80;font-weight:700;min-width:48px;text-align:right">{item['calories']}</span>
            </div>
          </td>
          <td style="padding:10px 14px;text-align:center;color:#80d8ff;font-size:13px">{item['protein_g']}g</td>
          <td style="padding:10px 14px;text-align:center;color:#b9f6ca;font-size:13px">{item['carbs_g']}g</td>
          <td style="padding:10px 14px;text-align:center;color:#ffe57f;font-size:13px">{item['fat_g']}g</td>
        </tr>"""

    summary_text = generate_summary({"items": items_out, "totals": totals})

    return f"""
<div style="font-family:'Segoe UI',system-ui,sans-serif;color:#f0f0f0;padding:4px 0">

  <!-- Score hero -->
  <div style="display:flex;align-items:center;gap:24px;padding:24px 28px;
              background:linear-gradient(135deg,#1a1040 0%,#0d1a2e 100%);
              border-radius:16px;margin-bottom:16px;border:1px solid rgba(108,99,255,0.3)">
    <div style="text-align:center;min-width:110px">
      <div style="font-size:56px;font-weight:800;color:{color};line-height:1;
                  text-shadow:0 0 40px {color}55">{s}</div>
      <div style="font-size:10px;letter-spacing:2px;color:#888;margin-top:4px">HEALTH SCORE</div>
    </div>
    <div style="width:1px;height:60px;background:rgba(255,255,255,0.1)"></div>
    <div style="flex:1">
      <div style="font-size:22px;font-weight:700;color:{color};letter-spacing:1px">{label}</div>
      <div style="font-size:13px;color:#aaa;margin-top:6px;line-height:1.6">{summary_text}</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:center;min-width:200px">
      <div style="background:rgba(255,138,128,0.1);border-radius:10px;padding:10px 8px;border:1px solid rgba(255,138,128,0.2)">
        <div style="font-size:20px;font-weight:800;color:#ff8a80">{totals['calories']}</div>
        <div style="font-size:10px;color:#888;letter-spacing:1px">KCAL</div>
      </div>
      <div style="background:rgba(128,216,255,0.1);border-radius:10px;padding:10px 8px;border:1px solid rgba(128,216,255,0.2)">
        <div style="font-size:20px;font-weight:800;color:#80d8ff">{totals['protein_g']}g</div>
        <div style="font-size:10px;color:#888;letter-spacing:1px">PROTEIN</div>
      </div>
      <div style="background:rgba(185,246,202,0.1);border-radius:10px;padding:10px 8px;border:1px solid rgba(185,246,202,0.2)">
        <div style="font-size:20px;font-weight:800;color:#b9f6ca">{totals['carbs_g']}g</div>
        <div style="font-size:10px;color:#888;letter-spacing:1px">CARBS</div>
      </div>
      <div style="background:rgba(255,229,127,0.1);border-radius:10px;padding:10px 8px;border:1px solid rgba(255,229,127,0.2)">
        <div style="font-size:20px;font-weight:800;color:#ffe57f">{totals['fat_g']}g</div>
        <div style="font-size:10px;color:#888;letter-spacing:1px">FAT</div>
      </div>
    </div>
  </div>

  <!-- Breakdown table -->
  <div style="background:#0f0f1a;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.08)">
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
          <th style="padding:12px 14px;text-align:left;color:#555;font-size:11px;letter-spacing:1.5px;font-weight:600">ITEM</th>
          <th style="padding:12px 14px;text-align:center;color:#555;font-size:11px;letter-spacing:1.5px;font-weight:600">PORTION</th>
          <th style="padding:12px 14px;text-align:right;color:#ff8a80;font-size:11px;letter-spacing:1.5px;font-weight:600">CALORIES</th>
          <th style="padding:12px 14px;text-align:center;color:#80d8ff;font-size:11px;letter-spacing:1.5px;font-weight:600">PROTEIN</th>
          <th style="padding:12px 14px;text-align:center;color:#b9f6ca;font-size:11px;letter-spacing:1.5px;font-weight:600">CARBS</th>
          <th style="padding:12px 14px;text-align:center;color:#ffe57f;font-size:11px;letter-spacing:1.5px;font-weight:600">FAT</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
      <tfoot>
        <tr style="border-top:1px solid rgba(255,255,255,0.08);background:rgba(108,99,255,0.08)">
          <td style="padding:13px 14px;font-weight:700;color:white;letter-spacing:0.5px;font-size:13px">TOTAL</td>
          <td style="padding:13px 14px;text-align:center;color:#444">—</td>
          <td style="padding:13px 14px;text-align:right;font-weight:800;color:#ff8a80;font-size:16px">{totals['calories']}</td>
          <td style="padding:13px 14px;text-align:center;font-weight:700;color:#80d8ff">{totals['protein_g']}g</td>
          <td style="padding:13px 14px;text-align:center;font-weight:700;color:#b9f6ca">{totals['carbs_g']}g</td>
          <td style="padding:13px 14px;text-align:center;font-weight:700;color:#ffe57f">{totals['fat_g']}g</td>
        </tr>
      </tfoot>
    </table>
  </div>
</div>"""

def analyse(image, threshold, use_yolo):
    if image is None:
        return None, "<p style='color:#555;padding:20px;font-family:Segoe UI'>Upload a tray photo to begin.</p>"

    img = Image.fromarray(image).convert("RGB")

    if use_yolo and YOLO is not None:
        raw = run_yolo(img, threshold)
    else:
        raw = run_resnet(img, threshold)

    if not raw:
        return img, "<p style='color:#555;padding:20px'>Nothing detected — try lowering the threshold.</p>"

    items_out, totals = [], {"calories":0.,"protein_g":0.,"carbs_g":0.,"fat_g":0.}
    for item in raw:
        cid = CATEGORIES.index(item["label"]) if item["label"] in CATEGORIES else -1
        n = estimate_nutrition(cid, item["grams"])
        entry = {"food":item["label"],"grams":item["grams"],
                 "calories":n.calories,"protein_g":n.protein_g,
                 "carbs_g":n.carbs_g,"fat_g":n.fat_g}
        if "confidence" in item: entry["confidence"] = item["confidence"]
        if "bbox" in item: entry["bbox"] = item["bbox"]
        items_out.append(entry)
        for k in totals: totals[k] += entry[k]

    totals = {k: round(v,1) for k,v in totals.items()}
    annotated = annotate(img, raw)
    html = build_output(items_out, totals, annotated)
    return annotated, html

# ── CSS ───────────────────────────────────────────────────────────
css = """
body { background: #08080f !important; }
.gradio-container {
    background: #08080f !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
footer { display: none !important; }

/* Upload box */
.gr-image { background: #0f0f1a !important; border: 1px solid rgba(108,99,255,0.3) !important; border-radius: 12px !important; }

/* Analyse button */
button.lg { background: #6C63FF !important; color: white !important; border: none !important;
            border-radius: 10px !important; font-weight: 700 !important; font-size: 16px !important;
            letter-spacing: 0.5px !important; transition: all 0.2s !important;
            box-shadow: 0 0 24px rgba(108,99,255,0.4) !important; }
button.lg:hover { background: #5a52e0 !important; transform: translateY(-1px) !important;
                  box-shadow: 0 0 36px rgba(108,99,255,0.6) !important; }

/* Accordion */
.gr-accordion { background: #0f0f1a !important; border: 1px solid rgba(255,255,255,0.08) !important;
                border-radius: 10px !important; }

/* Slider */
input[type=range] { accent-color: #6C63FF !important; }

/* Labels */
label > span { color: #666 !important; font-size: 11px !important; font-weight: 600 !important;
               text-transform: uppercase !important; letter-spacing: 1px !important; }
"""

yolo_ok = YOLO is not None

with gr.Blocks(title="Smart Tray") as demo:
    gr.HTML("""
    <div style="padding:32px 0 24px;text-align:center">
      <div style="display:inline-flex;align-items:center;gap:10px;
                  background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.4);
                  border-radius:999px;padding:6px 18px;font-size:12px;color:#9d97ff;
                  letter-spacing:1.5px;font-weight:600;margin-bottom:20px">
        AI · COMPUTER VISION · NUTRITION
      </div>
      <h1 style="font-size:48px;font-weight:800;color:white;margin:0 0 12px;
                 letter-spacing:-1.5px;line-height:1.1">
        Smart<span style="color:#6C63FF">Tray</span>
      </h1>
      <p style="color:#666;font-size:16px;margin:0;max-width:500px;margin:0 auto">
        Point a camera at any cafeteria tray. Get instant calorie and macro breakdowns — powered by computer vision.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            image_input = gr.Image(label="Upload Tray Photo", type="numpy", height=320)
            with gr.Accordion("⚙  Options", open=False):
                threshold = gr.Slider(0.1, 0.9,
                    value=CFG.inference.confidence_threshold, step=0.05,
                    label="Confidence Threshold")
                use_yolo = gr.Checkbox(
                    value=yolo_ok,
                    label=f"YOLO Detector  {'✓ active' if yolo_ok else '✗ unavailable'}",
                    interactive=yolo_ok)
            run_btn = gr.Button("⚡  Analyse Tray", variant="primary", size="lg")

        with gr.Column(scale=1, min_width=320):
            image_output = gr.Image(label="Detection", height=320)

    result_html = gr.HTML()

    run_btn.click(
        fn=analyse,
        inputs=[image_input, threshold, use_yolo],
        outputs=[image_output, result_html],
    )
    gr.Examples(
    examples=["examples/tray_1.jpg", "examples/tray_2.png",
              "examples/tray_3.png", "examples/tray_4.png"],
    inputs=image_input,
    label="Example Trays",
    )
    
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 8px;color:#333;font-size:12px;letter-spacing:0.5px">
      IE University · AI Project · Filo · Nicolas · Yago · Dimash · JP · Santi
    </div>""")


if __name__ == "__main__":
    demo.launch(share=True, css=css)