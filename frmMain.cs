using System;
using System.Linq;
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
            cmbInterval.Text = "Hourly";

            //dtpStartDate.Value = DateTime.Today.AddDays(-1);
        }

        private void UpdateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            frmUpdate updateForm = new frmUpdate();
            updateForm.ShowDialog();
        }

        #region Predict
        private string stockTicker;
        private string priceMetric;
        private string interval;
        private int[] startDate;
        private int[] endDate;

        private List<double> actualDataPoints = new List<double>();
        private List<double> predictedDataPoints = new List<double>();

        private void BtnPredict_Click(object sender, EventArgs e)
        {
            actualDataPoints.Clear();
            predictedDataPoints.Clear();

            stockTicker = txtStockTicker.Text.ToUpper();
            priceMetric = cmbPriceMetric.Text.ToLower();
            interval = cmbInterval.Text == "Hourly" ? "1m" : "1h";
            startDate = new int[] { dtpStartDate.Value.Year, dtpStartDate.Value.Month, dtpStartDate.Value.Day, dtpStartDate.Value.Hour, dtpStartDate.Value.Minute };
            endDate = new int[] { dtpEndDate.Value.Year, dtpEndDate.Value.Month, dtpEndDate.Value.Day, dtpEndDate.Value.Hour, dtpEndDate.Value.Minute };
            //MessageBox.Show(string.Join(",",startDate), "Start Date");
            //MessageBox.Show(string.Join(",", endDate), "End Date");

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

        private async Task PredictAsync()
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

                    //try
                    //{
                    dynamic actualData = predictComponent.get_actual_data(stockTicker, startDate, endDate, interval, priceMetric);

                    foreach (var x in actualData)
                        actualDataPoints.Add((double)x);

                    //}
                    //catch { MessageBox.Show(string.Format("The {0} stock cannot be found.", stockTicker), "Stock Prophet", MessageBoxButtons.OK, MessageBoxIcon.Warning); return; }

                    //try
                    //{
                    dynamic historicalData = predictComponent.historical_prediction(stockTicker, startDate, endDate, interval, priceMetric);

                    foreach (var x in historicalData)
                        predictedDataPoints.Add((double)x);
                    //}
                    //catch { MessageBox.Show(string.Format("A prediction cannot be generated for {0} for the time frame selected.", stockTicker), "Stock Prophet", MessageBoxButtons.OK, MessageBoxIcon.Warning); return; }
                }
            });
        }

        private bool IsFull()
        {
            return (txtStockTicker.Text != string.Empty && dtpStartDate.Value != null && dtpEndDate.Value != null && cmbPriceMetric.Text != string.Empty);
        }

        private void DrawChart()
        {
            chartStock.Series.Clear();
            
            ChartArea chartArea = new ChartArea();
            chartArea.AlignmentStyle = AreaAlignmentStyles.All;
            chartStock.ChartAreas.Add(chartArea);

            Series actual = new Series();
            actual.Name = "ActualData";
            actual.LegendText = "Actual Data";
            actual.ChartType = SeriesChartType.Line;

            Series predicted = new Series();
            predicted.Name = "PredictedData";
            predicted.LegendText = "Predicted Data";
            predicted.ChartType = SeriesChartType.Line;

            chartStock.Series.Add(actual);
            chartStock.Series.Add(predicted);

            var minActualData = actualDataPoints.Min();
            var maxActualData = actualDataPoints.Max();

            var minPredictedData = predictedDataPoints.Min();
            var maxPredictedData = predictedDataPoints.Max();

            var minPoint = Math.Min(minActualData, minPredictedData);
            var maxPoint = Math.Max(maxActualData, maxPredictedData);

            chartStock.ChartAreas[0].AxisY.Minimum = minPoint - 1;
            chartStock.ChartAreas[0].AxisY.Maximum = maxPoint + 1;

            chartStock.ChartAreas[0].AxisY.LabelStyle.Format = "{.00}";

            for (int i = 0; i < actualDataPoints.Count; i++)
                chartStock.Series["ActualData"].Points.AddXY(i + 1, actualDataPoints[i]);
            
            for (int i = 0; i < predictedDataPoints.Count ; i++)
                chartStock.Series["PredictedData"].Points.AddXY(i + 1, predictedDataPoints[i]);
            
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

        private void btnExpandChart_Click(object sender, EventArgs e)
        {
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            chartStock.Serializer.Save(ms);
            ms.Flush();

            frmChart ch = new frmChart(ms);
            ch.ShowDialog();
        }
    }
}
