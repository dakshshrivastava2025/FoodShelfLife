import cv2
from backend.cv_analyzer import BreadVisionAnalyzer
from backend.model import BreadShelfLifeModel

analyzer = BreadVisionAnalyzer()
model = BreadShelfLifeModel()

test_cases = [
    ('Fresh Day 1', r'bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261010_110000.jpg', 110, 75),
    ('Staling Day 2', r'bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261011_070000.jpg', 180, 130),
    ('Early Mold Day 3', r'bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261011_211708.jpg', 290, 210),
    ('Moderate Mold Day 4', r'bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261012_113417.jpg', 480, 370),
    ('Heavy Mold Day 5', r'bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261013_044251.jpg', 720, 560),
]

print("Multimodal Model Validation Across 5 Spoilage Stages:")
for label, img_path, mq135, mq2 in test_cases:
    img = cv2.imread(img_path)
    cv_res = analyzer.analyze_image(img)
    pred = model.predict(mq135, mq2, cv_res)
    cov = cv_res.get('mold_coverage_pct', 0.0)
    print(f"\n[{label}]")
    print(f"  Input: MQ-135={mq135}, MQ-2={mq2}, Visual Mold={cov}%")
    print(f"  Remaining Shelf Life: {pred['remaining_shelf_life_hours']}h ({pred['remaining_shelf_life_formatted']})")
    print(f"  Freshness Score:      {pred['freshness_score']}%")
    print(f"  Status Badge:         {pred['status_badge']}")
    print(f"  Food Recommendation:  {pred['recommendation']}")
