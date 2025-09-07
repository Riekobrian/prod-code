🚗 **Kenya Car Price Predictor**

A smart car price prediction app for the Kenyan market using machine learning and year-aware market intelligence.

## 🌟 Features

- **Year-Aware Predictions**: Uses hierarchical matching system considering manufacturing year
- **Market Intelligence**: Leverages real Kenyan car market data
- **Smart Defaults**: Automatically estimates insurance costs, engine specs, and other features
- **User-Friendly**: Simple interface with detailed explanations

## 🚀 Live Demo

Visit the live app: [Your App URL Here]

## 🛠️ Technology Stack

- **Backend**: Python, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud / Heroku / Railway

## 📊 How It Works

The system uses a hierarchical matching approach:

1. **Exact Match**: Same make, model, and year
2. **Year Range**: Same make/model in 5-year window
3. **Make/Model**: All years for this combination
4. **Make/Year**: Same manufacturer in year range
5. **Make Only**: All models from manufacturer
6. **Year Only**: All cars from similar years
7. **Global**: Market averages

## 🎯 Accuracy

The model achieves high accuracy by using **annual insurance costs** as the primary predictor (96%+ importance), which correlates strongly with car value in the Kenyan market.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

<!-- ---

# Procfile (for Heroku)
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0

--- -->