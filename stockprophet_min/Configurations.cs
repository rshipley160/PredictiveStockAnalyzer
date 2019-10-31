using System;

namespace StockProphet
{
    /* Configurations.cs - Used to set appropriate properties for application to run. Do not change unless Env folder is moved.
     * TODO: N/A
     */
    public class Configurations
    {
        private static string envPath = AppDomain.CurrentDomain.BaseDirectory + @"env\Scripts";
        private static string pythonPath = AppDomain.CurrentDomain.BaseDirectory + @"env\lib\site-packages;" + AppDomain.CurrentDomain.BaseDirectory + @"env\lib";

        /// <summary>
        /// Loads appropriate environment variables. Run once in beginning of application.
        /// </summary>
        public static void Load()
        {
            Environment.SetEnvironmentVariable("PATH", envPath, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONHOME", envPath, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath, EnvironmentVariableTarget.Process);
        }
    }
}
