using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static FirstAIConsoleApp.Modal.DataModel;

namespace SentimentAnalysis.HelperClass
{
    internal class Demo1
    {
        private MLContext _context;
        public Demo1(MLContext context)
        {
            _context = context;
        }

        public void DemoRun()
        {
            // Define the training data with correct labels (0 = negative, 1 = positive)
            var trainingData = new[]
            {
                new SentimentData { Text = "I love this!", Label = true },
                new SentimentData { Text = "This is terrible.", Label = false },
                new SentimentData { Text = "I am so happy!", Label = true },
                new SentimentData { Text = "I hate this.", Label = false }
            };

            var trainData = _context.Data.LoadFromEnumerable(trainingData);

            // Build the model pipeline without MapValueToKey and use the correct binary classification trainer
            var pipeline = _context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(_context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Create prediction engine
            var predictionEngine = _context.Model.CreatePredictionEngine<SentimentData, SentimentalPrediction>(model);

            // Test with a new input
            var input = new SentimentData { Text = "This is awful!" };
            //var input = new SentimentData { Text = "I am so excited!" };
            var prediction = predictionEngine.Predict(input);

            // Output the result
            Console.WriteLine($"Text: {input.Text}");
            Console.WriteLine($"Prediction (Positive or Negative): {(prediction.PredictedLabel ? "Positive" : "Negative")}");

        }
    }
}
