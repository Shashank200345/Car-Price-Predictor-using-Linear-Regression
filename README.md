# 🚗 Car Price Predictor
Have a look: https://car-price-pre.streamlit.app/

A beautiful and interactive web application built with Streamlit that predicts car prices based on various features like company, year, mileage, and fuel type.

## 🌟 Features

- **Interactive Web Interface**: Clean and modern UI with intuitive controls
- **Real-time Predictions**: Get instant car price predictions
- **Multiple Input Options**: 
  - Company selection from available brands
  - Year of purchase slider (2000-2024)
  - Kilometers travelled input
   - Fuel type selection (Petrol, Diesel)
- **Data Insights**: View statistics and trends from the dataset
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have these files in your directory:
   # - app.py
   # - Cleaned data.xls
   # - LinearRegressionModel.pkl
   # - requirements.txt
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

## 📊 How It Works

The application uses a Linear Regression model trained on car data to predict prices based on:

- **Company**: Car manufacturer (Maruti, Hyundai, Honda, Toyota, etc.)
- **Year**: Year of purchase (2000-2024)
- **Kilometers Driven**: Total distance covered by the car
- **Fuel Type**: Petrol or Diesel

### Model Training

The app automatically trains a new Linear Regression model using your data file (`Cleaned data.xls`) to avoid compatibility issues with the existing pickle file.

## 🎯 Usage

1. **Select Car Specifications**:
   - Choose the car company from the dropdown
   - Adjust the year of purchase using the slider
   - Enter the kilometers driven
   - Select the fuel type

2. **Get Prediction**:
   - Click the "🔮 Predict Price" button
   - View the predicted price in the main area

3. **Explore Insights**:
   - Check the data insights panel for additional information
   - View top companies and average prices by fuel type

## 📁 Project Structure

```
Carprice-Predictor/
├── app.py                    # Main Streamlit application
├── Cleaned data.xls         # Car dataset
├── LinearRegressionModel.pkl # Trained model (backup)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Technical Details

- **Framework**: Streamlit
- **Machine Learning**: scikit-learn (Linear Regression)
- **Data Processing**: pandas, numpy
- **Visualization**: Built-in Streamlit components
- **Styling**: Custom CSS for enhanced UI

## 🎨 Features in Detail

### Interactive Interface
- **Sidebar Controls**: Easy-to-use input controls for all car specifications
- **Real-time Updates**: Instant prediction updates when parameters change
- **Visual Feedback**: Color-coded metrics and styled components

### Data Analysis
- **Dataset Statistics**: Overview of available car data
- **Company Distribution**: Most common car brands in the dataset
- **Price Trends**: Average prices by fuel type

### Responsive Design
- **Mobile Friendly**: Works on all device sizes
- **Modern UI**: Clean, professional appearance
- **Accessibility**: Easy-to-read fonts and colors

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment
You can deploy this app to various cloud platforms:

1. **Streamlit Cloud**: Upload to GitHub and deploy directly
2. **Heroku**: Use the Procfile and deploy as a web app
3. **AWS/Azure/GCP**: Deploy as a containerized application

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Error**: 
   - The app automatically trains a new model from your data
   - Make sure `Cleaned data.xls` is in the same directory

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 📈 Future Enhancements

- [ ] Add more car features (transmission, engine type, etc.)
- [ ] Implement advanced ML models (Random Forest, XGBoost)
- [ ] Add data visualization charts
- [ ] Include price comparison features
- [ ] Add export functionality for predictions

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving the UI/UX
- Enhancing the machine learning model
- Adding more data sources

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ using Streamlit and Machine Learning

---

**Happy Predicting! 🚗💰**
