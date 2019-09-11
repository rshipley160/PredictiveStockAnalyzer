using System;
using IronPython.Hosting;

namespace CSharpShell
{
    class Program
    {
        static void Main(string[] args)
        {
            var ipy = Python.CreateRuntime();
            dynamic pythonFile = ipy.UseFile("test.py");

            // Passing data into a Python function
            object[] data = {
                "String 1",
                100,
                true,
                10.03,
                'c'
            };
            pythonFile.call(data);

            // Retrieving data from a getter function
            Console.WriteLine();
            string name = pythonFile.getName();
            Console.WriteLine(name);


            // Passing user-inputed data to a Python function
            Console.WriteLine();
            string msg = Console.ReadLine();
            Console.WriteLine(pythonFile.printMsg(msg));

            // Squaring number
            Console.WriteLine();
            Console.WriteLine(pythonFile.squareNum(100));

        }
    }
}
