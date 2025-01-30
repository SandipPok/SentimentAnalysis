using Microsoft.ML;
using SentimentAnalysis.HelperClass;
using static FirstAIConsoleApp.Modal.DataModel;

namespace AIConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            //Demo1 demo1 = new Demo1(context);
            //demo1.DemoRun();

            Demo2 demo2 = new Demo2(context);
            demo2.Run();
        }
    }
}