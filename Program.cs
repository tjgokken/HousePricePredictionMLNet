using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction;

public class Program
{
    private static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load Data
        var dataView =
            mlContext.Data.LoadFromTextFile<HouseData>("VirtualTownHouseDataset.csv", hasHeader: true,
                separatorChar: ',');

        // Define preprocessing pipeline (e.g., encoding and normalization)
        var preprocessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding("ZipCode")
            .Append(mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms", "ZipCode"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        // Gradient Descent (SDCA)
        RunSdcaRegression(mlContext, preprocessingPipeline, dataView);

        // Decision Tree (FastTree)
        RunFastTreeRegression(mlContext, preprocessingPipeline, dataView);

        // Neural Network
        RunNeuralNetworkPythonScript();
    }

    // Method for SDCA (Gradient Descent)
    public static void RunSdcaRegression(MLContext mlContext, IEstimator<ITransformer> preprocessingPipeline,
        IDataView dataView)
    {
        var pipeline = preprocessingPipeline.Append(mlContext.Regression.Trainers.Sdca(
            "Price", l2Regularization: 0.1f)); // L2 Regularization to prevent overfitting

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        var metrics = mlContext.Regression.Evaluate(predictions, "Price");
        Console.WriteLine("=== Gradient Descent (SDCA) Results ===");
        Console.WriteLine(
            $"R^2: {metrics.RSquared}, MAE: {metrics.MeanAbsoluteError.ToString("N2", CultureInfo.InvariantCulture)}");
        MakePrediction(mlContext, model);
    }

    // Method for FastTree (Decision Tree)
    public static void RunFastTreeRegression(MLContext mlContext, IEstimator<ITransformer> preprocessingPipeline,
        IDataView dataView)
    {
        var pipeline = preprocessingPipeline.Append(mlContext.Regression.Trainers.FastTree("Price"));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        var metrics = mlContext.Regression.Evaluate(predictions, "Price");
        Console.WriteLine("=== Decision Tree (FastTree) Results ===");
        Console.WriteLine(
            $"R^2: {metrics.RSquared}, MAE: {metrics.MeanAbsoluteError.ToString("N2", CultureInfo.InvariantCulture)}");
        MakePrediction(mlContext, model);
    }

    public static void RunNeuralNetworkPythonScript()
    {
        var predictor = new NeuralNetworkPredictorPythonScript();
        var result = predictor.PredictHousePrice(2000, 3, "12345");

        if (result != null)
            Console.WriteLine($"Predicted Price: {result.PredictedPrice:C}");
        // Additional metrics are already printed by the Python script output
    }

    public static void MakePrediction(MLContext mlContext, ITransformer model)
    {
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

        var houseSample = new HouseData { Size = 2000, Bedrooms = 3, ZipCode = "12345" };
        var prediction = predictionEngine.Predict(houseSample);

        Console.WriteLine($"Predicted price for house: {prediction.Price.ToString("C", CultureInfo.CurrentCulture)}\n");
    }
}

public class HousePricePrediction
{
    [ColumnName("Score")] public float Price { get; set; }
}