using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static FirstAIConsoleApp.Modal.DataModel;

namespace SentimentAnalysis.HelperClass
{
    internal class Demo2
    {
        private MLContext _context;

        public Demo2(MLContext context)
        {
            _context = context;
        }

        public void Run()
        {
            // Define a larger training dataset
            var trainingData = new[]
            {
                new SentimentData { Text = "I love this product, it's amazing!", Label = true },
                new SentimentData { Text = "This is the worst thing I've ever bought.", Label = false },
                new SentimentData { Text = "Very satisfied with the results.", Label = true },
                new SentimentData { Text = "Absolutely terrible, never buying again.", Label = false },
                new SentimentData { Text = "I'm so happy with my purchase!", Label = true },
                new SentimentData { Text = "Horrible customer service.", Label = false },
                new SentimentData { Text = "Best experience I've had!", Label = true },
                new SentimentData { Text = "I regret buying this.", Label = false },
                new SentimentData { Text = "I would recommend this to anyone!", Label = true },
                new SentimentData { Text = "Awful, don't waste your money.", Label = false },
                new SentimentData { Text = "I feel great after using it!", Label = true },
                new SentimentData { Text = "Worst purchase of my life.", Label = false }
            };

            // Split the data into training and testing sets (80% training, 20% testing)
            var trainTestSplit = _context.Data.TrainTestSplit(_context.Data.LoadFromEnumerable(trainingData), testFraction: 0.2);

            // Build the pipeline
            var pipeline = _context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(_context.BinaryClassification
                .Trainers
                .SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainTestSplit.TrainSet);

            // Evaluate the model on the test data
            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = _context.BinaryClassification.Evaluate(predictions);

            // Output the evaluation metrics
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:P2}");

            // Create prediction engine
            var predictionEngine = _context.Model.CreatePredictionEngine<SentimentData, SentimentalPrediction>(model);

            // Test with multiple input texts
            var testSentences = new[]
            {
                "I love this product!",
                "I will never buy this again.",
                "Fantastic experience, highly recommend!",
                "Waste of money, complete disappointment.",
                "This product is okay, not great but not bad.",
                "So happy I made this purchase."
            };

            Console.WriteLine("\nPrediction Results:\n");

            foreach (var sentence in testSentences)
            {
                var input = new SentimentData { Text = sentence };
                var prediction = predictionEngine.Predict(input);

                // Output the result for each test sentence
                Console.WriteLine($"Text: {input.Text}");
                Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Positive" : "Negative")}\n");
            }
        }
    }
}
