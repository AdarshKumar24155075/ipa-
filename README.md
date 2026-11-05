# Image Enhancement & Restoration System - Backend
Flask + OpenCV REST API.  `enhancer.py` = algorithms, `app.py` = API + serves `static/index.html`.

## Run
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5000
    # production: gunicorn -w 2 -b 0.0.0.0:8000 app:app

## API
POST /api/enhance  (multipart/form-data)
  image (file, required) | mask (file, optional, white = damaged)
  denoise (0-30) | deblur (sigma, 0=off) | contrast (CLAHE clip, 0=off) | gamma | sharpen (0-2)
  scale (1-4) | white_balance (true/false) | auto_scratch (true/false)
Returns JSON: { image: "data:image/png;base64,...", width, height }
GET /api/health

    curl -F image=@photo.jpg -F denoise=10 -F scale=2 http://127.0.0.1:5000/api/enhance

## Reproduce experiments
    python evaluate.py /path/to/Set14 --sigma 25 --scale 2
flask>=3.0
flask-cors>=4.0
opencv-python-headless>=4.9
numpy>=1.26
scipy>=1.11
scikit-image>=0.22
gunicorn>=21.2
