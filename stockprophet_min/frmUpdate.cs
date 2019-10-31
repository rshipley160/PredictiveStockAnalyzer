using System;
using System.Threading.Tasks;
using System.Windows.Forms;
using Python.Runtime;

namespace StockProphet
{
    public partial class frmUpdate : Form
    {
        public frmUpdate()
        {
            InitializeComponent();
        }

        private async void SetupDataCollectorAsync()
        {
            await SetupDataCollector();
        }

        private Task SetupDataCollector()
        {
            return Task.Run(() =>
            {
                using (Py.GIL())
                {
                    dynamic dataCollector = Py.Import("stock_components.sourceCode.data_collector");
                    dataCollector.DataCollector.setup();
                }
            });
        }

        private void FrmUpdate_Load(object sender, EventArgs e)
        {
            SetupDataCollectorAsync();
        }
    }
}
