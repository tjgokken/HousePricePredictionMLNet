using Microsoft.ML.Data;

namespace HousePricePrediction;

public class HouseData
{
    [LoadColumn(0)] // Maps to the first column in the CSV
    public float Size { get; set; }

    [LoadColumn(1)] // Maps to the second column in the CSV
    public float Bedrooms { get; set; }

    [LoadColumn(2)] // Maps to the third column in the CSV
    public float Price { get; set; }

    [LoadColumn(3)] // Maps to the fourth column in the CSV
    public required string ZipCode { get; set; }
}