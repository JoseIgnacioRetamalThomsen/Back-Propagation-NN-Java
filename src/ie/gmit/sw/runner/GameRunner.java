package ie.gmit.sw.runner;

import ie.gmit.sw.ai.nn.*;
import ie.gmit.sw.ai.nn.activator.*;

public class GameRunner {

	private double[][] data = { // Health, Sword, Gun, Enemies
			{ 2, 0, 0, 0 }, { 2, 0, 0, 1 }, { 2, 0, 1, 1 }, { 2, 0, 1, 2 }, { 2, 1, 0, 2 }, { 2, 1, 0, 1 },
			{ 1, 0, 0, 0 }, { 1, 0, 0, 1 }, { 1, 0, 1, 1 }, { 1, 0, 1, 2 }, { 1, 1, 0, 2 }, { 1, 1, 0, 1 },
			{ 0, 0, 0, 0 }, { 0, 0, 0, 1 }, { 0, 0, 1, 1 }, { 0, 0, 1, 2 }, { 0, 1, 0, 2 }, { 0, 1, 0, 1 } };

	private double[][] expected = { // Panic, Attack, Hide, Run
			{ 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 },
			{ 0.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 },
			{ 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 },
			{ 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 0.0 },
			{ 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 } };

	public GameRunner() throws Exception {
		NeuralNetwork nn = new NeuralNetwork(Activator.ActivationFunction.Sigmoid, 4, 3, 4);
		BackpropagationTrainer trainer = new BackpropagationTrainer(nn);
		trainer.train( Utils.normalize(data, 0, 2),expected, 0.6, 10000);

		int testIndex = 1;
		double[] t = {2.0,0.0,1.0,9000.0};
		double[] result = nn.process(t);
		
		//nn.process(data[testIndex]);
		for (int i = 0; i < expected[testIndex].length; i++) {
			System.out.print(expected[testIndex][i] + ",");
		}
		System.out.println("==>" + (Utils.getMaxIndex(result) + 1));
	}

	public static void main(String[] args) throws Exception {
		new GameRunner();
	}
}
