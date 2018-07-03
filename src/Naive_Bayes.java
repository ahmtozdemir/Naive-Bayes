import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class Naive_Bayes {

	public static void main(String[] args) {

		try {
			StringBuilder txtAreaShow = new StringBuilder();

			// arff file
			BufferedReader breader = null;
			breader = new BufferedReader(new FileReader("anneal.arff"));

			Instances train = new Instances(breader);
			train.setClassIndex(train.numAttributes() - 1);
			breader.close();

			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(train);

			Evaluation eval = new Evaluation(train);
			eval.crossValidateModel(nB, train, 10, new Random(1));

			System.out.println("Run Information\n=====================");
			System.out.println("Scheme: " + train.getClass().getName());
			System.out.println("Relation: ");

			System.out.println("\nClassifier Model(full training set)\n===============================");
			System.out.println(nB);

			System.out.println(eval.toSummaryString("\nSummary Results\n==================", true));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());

			// txtArea output
			txtAreaShow.append("\n\n\n");
			txtAreaShow.append("Run Information\n===================\n");
			txtAreaShow.append("Scheme: " + train.getClass().getName());

			txtAreaShow
					.append("\n\nClassifier Model(full training set)" + "\n======================================\n");
			txtAreaShow.append("" + nB);

			txtAreaShow.append(eval.toSummaryString("\n\nSummary Results\n==================\n", true));
			txtAreaShow.append(eval.toClassDetailsString());
			txtAreaShow.append(eval.toMatrixString());
			txtAreaShow.append("\n\n\n");

			System.out.println(txtAreaShow.toString());

		} catch (FileNotFoundException ex) {
			System.err.println("File not found");
			System.exit(1);
		} catch (IOException ex) {
			System.err.println("Invalid input or output.");
			System.exit(1);
		} catch (Exception ex) {
			System.err.println("Exception occured!");
			System.exit(1);
		}
	}
}
