using System.Diagnostics;
using System.Text;
using Newtonsoft.Json;

namespace HousePricePrediction;

public class NeuralNetworkPredictorPythonScript
{
    private readonly string _pythonPath;
    private readonly string _scriptPath;
    private readonly string _workingDirectory;

    public NeuralNetworkPredictorPythonScript(string pythonPath = "")
    {
        _pythonPath = !string.IsNullOrEmpty(pythonPath)
            ? pythonPath
            : Environment.GetEnvironmentVariable("EnvPython312") ?? @"C:/YourDefaultPythonPath/python.exe";
        _workingDirectory = Directory.GetCurrentDirectory();
        _scriptPath = Path.Combine(_workingDirectory, "HousePriceNeuralNetwork.py");
    }

    public PredictionResult? PredictHousePrice(double size, int bedrooms, string zipCode)
    {
        var processStartInfo = CreateProcessStartInfo(size, bedrooms, zipCode);

        try
        {
            using var process = Process.Start(processStartInfo);
            if (process == null)
            {
                LogError("Failed to start Python process.");
                return null;
            }

            var (output, errors) = GetProcessOutput(process);
            LogErrors(errors);

            if (string.IsNullOrEmpty(output))
            {
                LogError("No output received from Python script");
                return null;
            }

            return ParsePredictionOutput(output);
        }
        catch (Exception ex)
        {
            LogError($"Error running Python script: {ex.Message}");
            return null;
        }
    }

    private ProcessStartInfo CreateProcessStartInfo(double size, int bedrooms, string zipCode)
    {
        return new ProcessStartInfo
        {
            FileName = _pythonPath,
            Arguments = $"\"{_scriptPath}\" {size} {bedrooms} {zipCode}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _workingDirectory,
            StandardOutputEncoding = Encoding.ASCII,
            StandardErrorEncoding = Encoding.ASCII,
            EnvironmentVariables =
            {
                ["TF_CPP_MIN_LOG_LEVEL"] = "3",
                ["TF_ENABLE_ONEDNN_OPTS"] = "0",
                ["PYTHONIOENCODING"] = "ascii"
            }
        };
    }

    private (string output, string errors) GetProcessOutput(Process process)
    {
        var output = process.StandardOutput.ReadToEnd();
        var errors = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (output, errors);
    }

    private PredictionResult? ParsePredictionOutput(string output)
    {
        try
        {
            var (jsonPart, preJsonOutput) = ExtractJsonFromOutput(output);
            if (string.IsNullOrEmpty(jsonPart)) return null;

            var jsonResult = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonPart);
            if (jsonResult == null) return null;

            if (jsonResult.ContainsKey("error"))
            {
                LogError($"Error from Python script: {jsonResult["error"]}");
                return null;
            }

            if (!HasRequiredFields(jsonResult)) return null;

            // Print the metrics output first
            if (!string.IsNullOrEmpty(preJsonOutput)) Console.WriteLine(preJsonOutput);

            return new PredictionResult
            {
                PredictedPrice = Convert.ToDouble(jsonResult["predicted_price"]),
                R2 = Convert.ToDouble(jsonResult["r2"]),
                Mae = Convert.ToDouble(jsonResult["mae"])
            };
        }
        catch (JsonException ex)
        {
            LogError($"Failed to parse JSON output: {ex.Message}");
            LogError($"Raw output: {output}");
            return null;
        }
    }

    private (string jsonPart, string preJsonOutput) ExtractJsonFromOutput(string output)
    {
        var startIndex = output.IndexOf('{');
        var endIndex = output.LastIndexOf('}');

        if (startIndex < 0 || endIndex < 0 || endIndex <= startIndex)
        {
            LogError($"Could not find valid JSON in output: {output}");
            return (string.Empty, string.Empty);
        }

        var preJsonOutput = output.Substring(0, startIndex).Trim();
        var jsonPart = output.Substring(startIndex, endIndex - startIndex + 1);
        return (jsonPart, preJsonOutput);
    }

    private bool HasRequiredFields(Dictionary<string, object> jsonResult)
    {
        var requiredFields = new[] { "predicted_price", "r2", "mae" };
        var missingFields = requiredFields.Where(field => !jsonResult.ContainsKey(field)).ToList();

        if (missingFields.Any())
        {
            LogError($"Missing required fields in JSON response: {string.Join(", ", missingFields)}");
            return false;
        }

        return true;
    }

    private void LogError(string message)
    {
        Console.WriteLine($"Error: {message}");
    }

    private void LogErrors(string errors)
    {
        if (!string.IsNullOrEmpty(errors)) Console.WriteLine($"Python script errors: {errors}");
    }

    public class PredictionResult
    {
        public double PredictedPrice { get; set; }
        public double R2 { get; set; }
        public double Mae { get; set; }
    }
}