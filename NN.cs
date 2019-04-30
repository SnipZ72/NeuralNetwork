using System;

namespace NeuralNetwork
{
    public class NN
    {
	public static void Main()
	{
	    NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
	    nn.Randomize();

	    Node[] tf = new Node[2];
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
	    Console.WriteLine(nn.WeightedSum(ft));

	}
    }


    public class NeuralNetwork
    {
	public Node[] inputs;
	public Node[] hidden;
	public Node[] outputs;

	public Weight[] inHid;
	public Weight[] hidOut;
    
	public NeuralNetwork(int inputs, int hidden, int outputs)
	{
	    this.inputs = new Node[inputs];
	    this.hidden = new Node[hidden];
	    this.outputs = new Node[outputs];

	    inHid = new Weight[inputs*hidden];
	    hidOut = new Weight[hidden*outputs];

	    for(int i=0; i < this.inputs.Length; i++)
	    {
		this.inputs[i] = new Node();
	    }
	
	    for(int i=0; i < this.hidden.Length; i++)
	    {
		this.hidden[i] = new Node();
	    }
	
	    for(int i=0; i < this.outputs.Length; i++)
	    {
		this.outputs[i] = new Node();
	    }
	
	    for(int i=0; i < this.inHid.Length; i++)
	    {
		this.inHid[i] = new Weight();
	    }
	}

	public void Randomize()
	{
	    for(int i=0; i < inHid.Length; i++)
		inHid[i].Randomize();
	}

	public void SupervisedTrain(Node[] inputs, double answer)
	{
	    double g = WeightedSum(inputs);

	    double error = answer - g;

	    for(int i=0; i < inHid.Length; i++)
		inHid[i].value += error;
	}

	public double WeightedSum(Node[] inputs)
	{
	    double sum = 0;

	    for(int i=0; i < inHid.Length; i++)
		sum += inputs[i].value * inHid[i].value + 1;

	    return Activate(sum);
	}

	public double Activate(double x)
	{
	    return 1/(1+Math.Pow(Math.E, -x));
	}
    }

    public class Node
    {
	public double value;

	public Node()
	{

	}
    
	public Node(double v)
	{
	    value = v;
	}

	public double WeightedSum()
	{
	    double sum = 0;
	    for(int i=0; i < weights.Length; i++)
		sum += value * weights[i].value + 1;

	    return sum;
	}

	public void RandomizeWeights()
	{
	    for(int i=0; i < weights.Length; i++)
		weights[i].Randomize();
	}

	public void Train(double error)
	{
	    for(int i=0; i < weights.Length; i++)
	    {
		weights[i].value += 0.01 * error;
	    }
	}
    }

    public class Weight
    {
	public double value;
    
	public Weight()
	{
	
	}

	public void Randomize()
	{
	    Random r = new Random();
	    value = (r.NextDouble()*2)-1;
	}
    }
}
