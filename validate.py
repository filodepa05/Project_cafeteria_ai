"""Quick validation script - checks if your repo is set up correctly."""

def main():
    print("\n" + "="*60)
    print("  Smart Tray Repository Check")
    print("="*60 + "\n")
    
    errors = []
    
    # Check 1: Can we import the dataset?
    print("Checking dataset module...")
    try:
        from src.dataset import CATEGORIES, NUM_CLASSES
        from src.nutrition import _DB
        
        print(f"   Food classes defined: {len(CATEGORIES)}")
        print(f"   NUM_CLASSES = {NUM_CLASSES}")
        print(f"   Nutrition entries = {len(_DB)}")
        
        if len(CATEGORIES) != len(_DB):
            print(f"   ERROR: CATEGORIES has {len(CATEGORIES)} items but nutrition table has {len(_DB)}")
            errors.append("Category/nutrition mismatch")
        
        if len(CATEGORIES) != NUM_CLASSES:
            print(f"   ERROR: CATEGORIES has {len(CATEGORIES)} items but NUM_CLASSES = {NUM_CLASSES}")
            errors.append("NUM_CLASSES mismatch")
            
    except Exception as e:
        print(f"   ERROR: Cannot import dataset: {e}")
        errors.append("Dataset import failed")
    
    print()
    
    # Check 2: Can we load the config?
    print("Checking config files...")
    try:
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        print(f"   base.yaml loads successfully")
        print(f"   Config num_classes = {cfg.model.num_classes}")
        
        if cfg.model.num_classes != len(CATEGORIES):
            print(f"   ERROR: Config has num_classes={cfg.model.num_classes} but CATEGORIES has {len(CATEGORIES)}")
            errors.append("Config mismatch")
            
    except Exception as e:
        print(f"   ERROR: Cannot load config: {e}")
        errors.append("Config load failed")
    
    print()
    
    # Check 3: Can we create the model?
    print("Checking model...")
    try:
        import torch
        from src.models.tray_model import TrayModel
        from src.config import ModelConfig
        
        cfg = ModelConfig(backbone="resnet18", pretrained=False, num_classes=43, portion_hidden=32)
        model = TrayModel(cfg)
        print(f"   Model created successfully")
        
        # Try a forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        print(f"   Forward pass works")
        print(f"   Output shape: logits={out['logits'].shape}, grams={out['grams'].shape}")
        
    except Exception as e:
        print(f"   ERROR: Model test failed: {e}")
        errors.append("Model failed")
    
    print()
    
    # Summary
    print("="*60)
    if errors:
        print(f"FAILED - Found {len(errors)} error(s):")
        for e in errors:
            print(f"   {e}")
        print("\nFix these before continuing!")
        return 1
    else:
        print(" ALL CHECKS PASSED!")
        print("\nYour repo is ready. Next steps:")
        print("  1. Run tests: pytest tests/ -v")
        print("  2. Quick train: python train.py --config configs/experiment/debug.yaml")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
