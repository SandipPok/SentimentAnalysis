namespace FirstAIConsoleApp.Modal
{
    internal class DataModel
    {
        // SentimentData model with text and label
        public class SentimentData
        {
            public string Text { get; set; }
            public bool Label { get; set; }
        }

        // Prediction result class
        public class SentimentalPrediction
        {
            public bool PredictedLabel { get; set; }
        }
    }
}