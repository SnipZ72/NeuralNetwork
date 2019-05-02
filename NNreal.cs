using System;

namespace NeuralNetwork
{
    public class NN
    {
	public static void Main()
	{
	    NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
	    //nn.Randomize();

	    double[] tf = new double[2];
	    tf[0] = 1;
	    tf[1] = 0;

	    double[] ft = new double[2];
	    tf[0] = 0;
	    tf[1] = 1;

	    double[] ff = new double[2];
	    tf[0] = 0;
	    tf[1] = 0;

	    double[] tt = new double[2];
	    tf[0] = 1;
	    tf[1] = 1;
	    
	    double[] tfans = new double[1];
	    tfans[0] = 1;

	    double[] ftans = new double[1];
	    ftans[0] = 1;

	    double[] ttans = new double[1];
	    ttans[0] = 0;

	    double[] ffans = new double[1];
	    ffans[0] = 0;
	    
	    for(int i=0; i < nn.Guess(tf).GetLength(0); i++)
	    {
		for(int j=0; j < nn.Guess(tf).GetLength(1); j++)
		{
		    Console.WriteLine(nn.Guess(tf)[i,j]);
		    Console.WriteLine(nn.Guess(ft)[i,j]);
		    Console.WriteLine(nn.Guess(tt)[i,j]);
		    Console.WriteLine(nn.Guess(ff)[i,j]);
		}
	    }

	    Console.WriteLine("TRAINING...");

	    for(int i=0; i < 1000; i++)
	    {
		nn.Train(tf, tfans);
		nn.Train(ft, ftans);
		nn.Train(tt, ttans);
		nn.Train(ff, ffans);
	    }

	    Console.WriteLine("TRAINED");

	    for(int i=0; i < nn.Guess(tf).GetLength(0); i++)
	    {
		for(int j=0; j < nn.Guess(tf).GetLength(1); j++)
		{
		    Console.WriteLine(nn.Guess(tf)[i,j]);
		    Console.WriteLine(nn.Guess(ft)[i,j]);
		    Console.WriteLine(nn.Guess(tt)[i,j]);
		    Console.WriteLine(nn.Guess(ff)[i,j]);
		}
	    }
	    
	    /*Node[] tf = new Node[2];
	    tf[0] = new Node(0);
	    tf[1] = new Node(1);

	    Node[] tt = new Node[2];
	    tt[0] = new Node(1);
	    tt[1] = new Node(1);

	    Node[] ff = new Node[2];
	    ff[0] = new Node(0);
	    ff[1] = new Node(0);

	    Node[] ft = new Node[2];
	    ft[0] = new Node(1);
	    ft[1] = new Node(0);

	    Console.WriteLine(nn.WeightedSum(tf));
	    Console.WriteLine(nn.WeightedSum(tt));
	    Console.WriteLine(nn.WeightedSum(ff));
	    Console.WriteLine(nn.WeightedSum(ft));

	    Console.WriteLine("TRAINING...");
	
	    for(int i=0; i < 1000; i++)
	    {
		nn.SupervisedTrain(tf, 1);
		nn.SupervisedTrain(tt, 0);
		nn.SupervisedTrain(ff, 0);
		nn.SupervisedTrain(ft, 1);
	    }

	    Console.WriteLine("TRAINED.");
	
	    Console.WriteLine(nn.WeightedSum(tf));
	    Console.WriteLine(nn.WeightedSum(tt));
	    Console.WriteLine(nn.WeightedSum(ff));
	    Console.WriteLine(nn.WeightedSum(ft));*/

	}
    }


    public class NeuralNetwork
    {
	int inputNodes;
	int hiddenNodes;
	int outputNodes;

        double[,] wInHid;
        double[,] wHidOut;

	double learningRate;
    
	public NeuralNetwork(int inputs, int hidden, int outputs, double lr = 0.03)
	{

	    Random rand = new Random();
	    
	    inputNodes = inputs;
	    hiddenNodes = hidden;
	    outputNodes = outputs;
	    
	    wInHid = new double[hiddenNodes, inputNodes];
	    wHidOut = new double[outputNodes, hiddenNodes];

	    learningRate = lr;
	    
	    for(int i=0; i < wInHid.GetLength(0); i++)
	    {
		for(int j=0; j < wInHid.GetLength(1); j++)
		{
		    wInHid[i,j] = Gaussian(0.0, Math.Pow(inputNodes, -0.5), rand);
		}
	    }

	    for(int i=0; i < wHidOut.GetLength(0); i++)
	    {
		for(int j=0; j < wHidOut.GetLength(1); j++)
		{
		    wHidOut[i,j] = Gaussian(0.0, Math.Pow(hiddenNodes, -0.5), rand);
		}
	    }
	    
	}

	public double Gaussian(double mean, double standardDeviation, Random rand)
	{
	    double u1 = 1.0-rand.NextDouble();
	    double u2 = 1.0- rand.NextDouble();
	    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
	    double randNormal = mean + standardDeviation * randStdNormal;

	    return randNormal;
	}

	public double Sigmoid(double x)
	{
	    return 1/(1+Math.Pow(Math.E, -x));
	}

	public double[,] Guess(double[] inputs)
	{
	    //	    double[,] inputsMatrix = Transpose(ToMultiArray(inputs));
	    double[,] inputsMatrix = ToMultiArray(inputs);

	    double[,] hiddenInputs = Dot(wInHid, inputsMatrix);
	    
	    double[,] hiddenOutputs = SigmoidArray(hiddenInputs);

	    double[,] finalInputs = Dot(wHidOut, hiddenOutputs);

	    double[,] finalOutputs = SigmoidArray(finalInputs);
	    
	    return finalOutputs;
	}

	public void Train(double[] inputs, double[] targets)
	{
	    double[,] targetMA = ToMultiArray(targets);

	    //REDO Guess -- However we do use some of these variables so we can't just call Guess with inputs
	    double[,] inputsMatrix = ToMultiArray(inputs);

	    double[,] hiddenInputs = Dot(wInHid, inputsMatrix);
	    
	    double[,] hiddenOutputs = SigmoidArray(hiddenInputs);

	    double[,] finalInputs = Dot(wHidOut, hiddenOutputs);

	    double[,] finalOutputs = SigmoidArray(finalInputs);
	    //End of Guess

	    double[,] outputErrors = Subtract(targetMA, finalOutputs);
		    
	    double[,] hiddenErrors = Dot(Transpose(wHidOut), outputErrors);

	    //Console.WriteLine("Started Updating Input-Hidden Weights");
	    
	    for(int i=0; i < wInHid.GetLength(0); i++)
	    {
		for(int j=0; j < wInHid.GetLength(1); j++)
		{
		    //wInHid[i] += learningRate * Dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), inputsMatrix);
		    wInHid[i,j] += Value(Multiply(Dot((Multiply(Multiply(hiddenErrors, hiddenOutputs), Subtract(hiddenOutputs, 1.0))), Transpose(inputsMatrix)), learningRate));
		    //wInHid[i,j] = Add(Multiply(Dot((Multiply(Multiply(hiddenErrors, hiddenOutputs), Subtract(hiddenOutputs, 1.0))), inputsMatrix), learningRate), wInHid[i,j]);
		}
	    }

	    //Console.WriteLine("Finished Updating Input-Hidden Weights");
	    
	   // Console.WriteLine("Started Updating Hidden-Output Weights");
	    
	    for(int i=0; i < wHidOut.GetLength(0); i++)
	    {
		for(int j=0; j < wHidOut.GetLength(1); j++)
		{
		    wHidOut[i,j] += Value(Multiply(Dot((Multiply(Multiply(outputErrors, finalOutputs), Subtract(finalOutputs, 1.0))), Transpose(hiddenOutputs)), learningRate));
		}
	    }

	   // Console.WriteLine("Finished Updating Hidden-Output Weights\nFinished Training");
	}

	public double Value(double[,] a)
	{
	    double sum = 0;

	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    sum += a[i,j];
		}
	    }	    
	    
	    return sum;
	    
	}

	public double[,] ToMultiArray(double[] array)
	{
	    double[,] multiArray = new double[array.Length, 1];

	    for(int i=0; i < array.Length; i++)
	    {
		multiArray[i,0] = array[i];
	    }

	    return multiArray;
	}

	public double[,] Transpose(double[,] matrix)
	{
	    int r = matrix.GetLength(0);
	    int c = matrix.GetLength(1);
	    
	    double[,] transposedArray = new double[c,r];
	    
	    for(int i=0; i < r; i++)
	    {
		for(int j=0; j < c; j++)
		{
		    transposedArray[j,i] = matrix[i,j];
		}
	    }

	    return transposedArray;
	}

	public double[,] Dot(double[,] a, double[,] b)
	{
	    if(a.GetLength(1) != b.GetLength(0))
	    {
		Console.WriteLine("ERROR: DOT a columns {0} not equal to b rows {1}", a.GetLength(1), b.GetLength(0));
		return null;
	    }
	    
	    int r = a.GetLength(0);
	    int c = b.GetLength(1);

	    double[,] dotArray = new double[r,c];
	    
	    for(int i=0; i < r; i++)
	    {
		for(int j=0; j < c; j++)
		{
		    double sum = 0;
		    for(int k=0; k < a.GetLength(1); k++)
		    {
			sum += a[i,k] * b[k,j];
		    }
		    
		    dotArray[i,j] = sum;
		}
	    }

	    return dotArray;
	}

	public double[,] SigmoidArray(double[,] x)
	{
	    int r = x.GetLength(0);
	    int c = x.GetLength(1);

	    double[,] output = new double[r,c];
	    
	    for(int i=0; i < r; i++)
	    {
		for(int j=0; j < c; j++)
		{
		    output[i,j] = Sigmoid(x[i,j]);
		}
	    }

	    return output;
	}

	public double[,] Multiply(double[,] a, double[,] b)
	{
	    int r = a.GetLength(0);
	    int c = a.GetLength(1);
	    double[,] output = new double[r,c];

	    if(r != b.GetLength(0) && c != b.GetLength(1))
	    {
		Console.WriteLine("ERROR: MULTIPLY a size [{0},{1}] not equal to b size [{2},{3}]\n", r, c, b.GetLength(0), b.GetLength(1));
		return null;
	    }
	    
	    for(int i=0; i < r; i++)
	    {
		for(int j=0; j < c; j++)
		{
		    output[i,j] = a[i,j] - b[i,j];
		}
	    }

	    return output;
	    
	}

	public double[,] Multiply(double[,] a, double b)
	{
	    double[,] output = new double[a.GetLength(0), a.GetLength(1)];
	    
	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    output[i,j] = a[i,j] * b;
		}
	    }

	    return output;
	}

	public double[,] Add(double[,] a, double[,] b)
	{
	    if(a.GetLength(0) != b.GetLength(0) &&
	       a.GetLength(1) != b.GetLength(1))
	    {
		Console.WriteLine("ERROR: Add a size [{0},{1}] not equal to b size [{2},{3}]\n", a.GetLength(0), a.GetLength(1), b.GetLength(0), b.GetLength(1));
		return null;
	    }

	    double[,] output = new double[a.GetLength(0), a.GetLength(1)];
	    
	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    output[i,j] = a[i,j] + b[i,j];
		}
	    }
	    
	    return output;
	}

	public double[,] Add(double[,] a, double b)
	{
	    double[,] output = new double[a.GetLength(0), a.GetLength(1)];
	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    output[i,j] = a[i,j] + b;
		}
	    }

	    return output;
	}
	
	public double[,] Subtract(double[,] a, double[,] b)
	{
	    if(a.GetLength(0) != b.GetLength(0) &&
	       a.GetLength(1) != b.GetLength(1))
	    {
		Console.WriteLine("ERROR: Subtract a size [{0},{1}] not equal to b size [{2},{3}]\n", a.GetLength(0), a.GetLength(1), b.GetLength(0), b.GetLength(1));
		return null;
	    }

	    double[,] output = new double[a.GetLength(0), a.GetLength(1)];
	    
	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    output[i,j] = a[i,j] - b[i,j];
		}
	    }
	    
	    return output;
	}

	public double[,] Subtract(double[,] a, double b)
	{
	    double[,] output = new double[a.GetLength(0), a.GetLength(1)];
	    for(int i=0; i < a.GetLength(0); i++)
	    {
		for(int j=0; j < a.GetLength(1); j++)
		{
		    output[i,j] = a[i,j] - b;
		}
	    }

	    return output;
	}

	
	/*public double Dot(double[,] a, double[,] b)
	{
	    int r = a.GetLength(0);
	    int c = b.GetLength(1);

	    double dot = 0;
	    
	    for(int i=0; i < r; i++)
	    {
		for(int j=0; j < c; j++)
		{
		    dot += a[i,j] * b[i,j];
		}
	    }

	    return dot;
	}*/
    }
}
