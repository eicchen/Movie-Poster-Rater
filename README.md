# Movie Poster Bayesian Net

> *"What can you infer about a movie by looking at its poster?"*
> 
> <p style="text-align: right">—Softmax Squad (Eric Chen, Morgan Klutzke, Asher Silverglade)</p>


Predicts movie genres and rating buckets from a poster image using a feature-extraction pipeline and Bayesian networks.

## Highlights
+ Poster **feature extraction**: color, texture, layout, lighting, blur, negative space.
+ **Visual content signals**: face detection (MediaPipe), object detection (YOLOv8), OCR text features (EasyOCR).
+ **Bayesian network inference** for genre and rating predictions.
+ **Counterfactual analysis** helpers to suggest minimal feature changes to shift predicted genre or rating.

## Repository Layout
```
.
├── data/
│   ├── genres.json                       # Metadata used in notebooks
│   ├── keywords.json                     # Metadata used in notebooks
│   ├── movies.parquet                    # Metadata used in notebooks
│   ├── poster_genre_bn.pkl
│   ├── poster_genre_bn_bundle.pkl        # Bayesian network bundle for genre
│   ├── poster_rating_bn_bundle.pkl       # Bayesian network bundle for rating
│   └── yolov8n.pt                        # YOLOv8n weights used for object features
├── notebooks/
│   ├── BayesianTrainer.ipynb             # Bayesian network training
│   ├── FeatureIdentifier.ipynb           # Exploratory feature work
│   ├── PosterPicker.ipynb                # Poster curation / selection workflow
│   └── SecondBayesianTrainer.ipynb       # Bayesian network training
├── src/
│   ├── ImageToGenre_ClearnInterface.py   # Core feature extraction + inference API
│   ├── __init__.py
│   ├── api.py
│   ├── fetch_posters.py
│   └── main.py
├── .gitignore
├── environment.yaml                      # Conda environment definition
├── main.ipynb
└── README.md
```

