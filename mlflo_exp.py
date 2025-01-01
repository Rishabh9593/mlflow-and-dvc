import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment Setup
mlflow.set_experiment("load_diabetes")

# Log an experiment run
with mlflow.start_run():
    # Define model and train it
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("features", X_train.shape[1])

    # Log model coefficients (as an example of metrics)
    mlflow.log_metric("intercept", model.intercept_)
    mlflow.log_metric("score", model.score(X_test, y_test))

    # Log the model itself
    mlflow.sklearn.log_model(model, "linear_regression_model")

    # Optionally save model artifacts
    #mlflow.log_artifact('model_output.txt')

with mlflow.start_run():
    # Experiment 2: Try a different model or hyperparameter
    # Create a pipeline with StandardScaler and LinearRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Log experiment details
    mlflow.log_param("model", "LinearRegression (Normalized)")
    mlflow.log_metric("score", pipeline.score(X_test, y_test))
    mlflow.sklearn.log_model(pipeline, "normalized_linear_regression_model")


