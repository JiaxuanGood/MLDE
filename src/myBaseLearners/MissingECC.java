package myBaseLearners;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.MultiLabelInstances;
import myStack.EvaluationsMul;
import myStack.Mat;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class MissingECC {
	public static double[] singleLearner_ECC(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		SMO basiclearner = new SMO();
		basiclearner.setKernel(new RBFKernel());
		EnsembleOfClassifierChains ECC = new EnsembleOfClassifierChains(basiclearner,10,true,true);
		ECC.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = ECC.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static void main(String[] args) throws Exception {
		//data prepare
		System.out.println("ALL BEGIN: MulLabelLearner");
		String[] filename = {"CHD_49","Emotions","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","HumanGO",
				"Image","Langlog","Scene","Slashdot","Chess","Water_quality","Yeast","Yelp"};
		for(int testTimes = 0; testTimes<20; testTimes++) {
			for(int fileIndex = 0; fileIndex < 16; fileIndex++) {
				String dataFileName = filename[fileIndex];
				
				String path = "E:\\multiLabel\\DATA\\arff-partitioned\\";
				path += dataFileName;
				path += "\\";
				
//				String arffFile_train = path + dataFileName + "-train.arff";
				String arffFile_train = "E:\\multiLabel\\Missing1\\modifiedTrainData_MAXIDE\\3\\" + dataFileName + 0 + ".arff";
				String arffFile_test = path + dataFileName + "-test.arff";
				String xmlFile = path + dataFileName + ".xml";
				MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
				MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
				
				int dimTrain = origData_train.getNumInstances();
				int dimTest = origData_test.getNumInstances();
				System.out.println(dataFileName + ": " + dimTrain + ":" + dimTest + ":" + origData_train.getNumLabels());
				
				double[] result;
				EvaluationsMul evaluations = new EvaluationsMul();
				result = MulLabelLearner.singleLearner_ECC(origData_train, origData_test);
				evaluations.prtConclusion(result, dataFileName, "ECC");
			}
		}
	}
}
