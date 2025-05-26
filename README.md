# 🔍 Face Recognition System

Real-time face identification using multiple reference photos for better accuracy.

## Setup & Run

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add Photos
Create folders with person names:
```
imgs/
├── alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── bob/
│   ├── img1.jpg
│   └── img2.jpg
└── charlie/
    ├── pic1.jpg
    └── pic2.jpg
```

### 3. Run
```bash
python pose_invariant_tracker.py
```

## Controls
- `q` - Quit
- `r` - Reload photos
- `+/-` - Adjust sensitivity

## Results
- **Green box** = Recognized person with name on top of the box
- **Red box** = Unknown face