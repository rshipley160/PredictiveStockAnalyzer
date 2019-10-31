using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Forms.DataVisualization.Charting;
using System.Windows.Forms;
using System.Drawing;
using Python.Runtime;

namespace StockProphet
{
    public partial class frmMain : Form
    {
        private event EventHandler<StatusUpdateEventsArgs> statusUpdate;

        public frmMain()
        {
            InitializeComponent();

            Configurations.Load();
            Runtime.PythonVersion = Version.Parse("3.7.4");
            Runtime.PythonDLL = Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Process) + @"\python37.dll";

            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();

            statusUpdate += FrmMain_statusUpdate;
        }


        private void FrmMain_Load(object sender, EventArgs e)
        {
            cmbPriceMetric.Text = "CLOSE";
        }

        private void UpdateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            frmUpdate updateForm = new frmUpdate();
            updateForm.ShowDialog();
        }

        #region Predict
        private string stockTicker;
        private string priceMetric;
        private List<double> actualDataPoints = new List<double>();
        private List<double> predictedDataPoints = new List<double>();

        private void BtnPredict_Click(object sender, EventArgs e)
        {
            stockTicker = txtStockTicker.Text;
            priceMetric = cmbPriceMetric.Text.ToLower();

            if (IsFull())
            {
                lblStockName.Text = "Stock Name: " + stockTicker;
                OnStatusUpdate("Running " + stockTicker + ". This may take a minute...");
                PredictAsync();
            }
            else
            {
                MessageBox.Show("All values need to be filled in.", "Stock Prophet", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private async void PredictAsync()
        {
            await PredictHistorical();

            OnStatusUpdate("Finished prediction.");
            DrawChart();
        }

        private Task PredictHistorical()
        {
            return Task.Run(() =>
            {
                using (Py.GIL())
                {
                    dynamic predictComponent = Py.Import("stock_components.sourceCode.stock_predictor");
                    dynamic actualData = predictComponent.get_actual_data(stockTicker, new double[] { 2019, 10, 28, 9, 30 }, new double[] { 2019, 10, 29, 15, 30 }, "1m", priceMetric);

                    foreach (var x in actualData)
                        actualDataPoints.Add((double)x);
                }
            });
        }

        private bool IsFull()
        {
            return (txtStockTicker.Text != string.Empty && dtpStartDate.Value != null && dtpEndDate.Value != null && cmbPriceMetric.Text != string.Empty);
        }

        private void DrawChart()
        {
            Series actual = new Series();
            actual.LegendText = "Actual Data";
            actual.ChartType = SeriesChartType.Line;

            Series predicted = new Series();
            predicted.LegendText = "Predicted Data";
            predicted.ChartType = SeriesChartType.Line;

            foreach (double d in actualDataPoints)
                actual.Points.Add(d);

            predicted.Points.Add(new double[] { 20, 30, 40, 50, 60 });

            chartStock.Series.Add(actual);
            chartStock.Series.Add(predicted);
        }

        #endregion

        #region Status

        protected void OnStatusUpdate(string status)
        {
            statusUpdate?.Invoke(this, new StatusUpdateEventsArgs() { Status = status });
        }
        private void FrmMain_statusUpdate(object sender, StatusUpdateEventsArgs e)
        {
            tslStatus.Text = "Status: " + e.Status;
        }

        #endregion
    }
}
