package myBaseLearners;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.meta.RAkELd;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import myStack.EvaluationsMul;
import myStack.Mat;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MulLabelLearner {
	public static double[] singleLearner_MLKNN(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		MLkNN mLkNN = new MLkNN();
		mLkNN.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = mLkNN.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_RAkEL(MultiLabelInstances traindata, MultiLabelInstances testdata, int basic) throws Exception {
		long t0 = System.currentTimeMillis();
		RAkELd mRAkEL;
		if(basic==0) {
			mRAkEL = new RAkELd(new BinaryRelevance(new SMO()));
		}
		else if (basic==1) {
			LabelPowerset mLabelPowerset = new LabelPowerset(new J48());
			mRAkEL = new RAkELd(mLabelPowerset);
		}
		else if (basic==2) {
			LabelPowerset mLabelPowerset = new LabelPowerset(new SMO());
			mRAkEL = new RAkELd(mLabelPowerset);
		}
		else {
			mRAkEL = new RAkELd();
		}
		mRAkEL.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = mRAkEL.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_RAkELo(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		RAkEL mRAkEL;
		SMO smo = new SMO();
		smo.setKernel(new RBFKernel());
		LabelPowerset mLabelPowerset = new LabelPowerset(smo);
		mRAkEL = new RAkEL(mLabelPowerset);
		mRAkEL.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = mRAkEL.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_RAkEL(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		RAkEL mRAkEL = new RAkEL();
		mRAkEL.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = mRAkEL.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_RAkELd(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		RAkELd mRAkELd = new RAkELd();
		mRAkELd.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = mRAkELd.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_AdaBoostMH(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		AdaBoostMH adaBoostMH = new AdaBoostMH();
		adaBoostMH.build(traindata);
		long t1 = System.currentTimeMillis();
		Instances testInstances = testdata.getDataSet();
		MultiLabelOutput[] outputs = new MultiLabelOutput[testdata.getNumInstances()];
		for(int i=0; i<testdata.getNumInstances(); i++) {
			outputs[i] = adaBoostMH.makePrediction(testInstances.get(i));
		}
		double[] times = {(t1-t0), (System.currentTimeMillis()-t1)};
		double[] result = new EvaluationsMul().evaluating(outputs, testdata);
		return Mat.vectcat(result, times);
	}
	
	public static double[] singleLearner_ECC(MultiLabelInstances traindata, MultiLabelInstances testdata) throws Exception {
		long t0 = System.currentTimeMillis();
		EnsembleOfClassifierChains ECC = new EnsembleOfClassifierChains();
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
		
//		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CHD_49","Emotions","Flags","Foodtruck",
//				"GnegativePseAAC","GpositiveGO","GpositivePseAAC","Image","PlantPseAAC","Scene","VirusGO","VirusPseAAC","Water_quality","Yeast",
//				"EukaryotePseAAC","Genbase","GnegativeGO","HumanPseAAC","Medical","PlantGO","Yelp",
//				"HumanGO","CAL500", "Coffee","tmc2007_500"};
//		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CAL500","CHD_49","Emotions",
//	        "Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","HumanGO","Image","Medical",
//	        "PlantGO","Scene","Coffee","tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"};
//		String[] filename = {"Birds","CHD_49","Corel5k","Emotions","Enron","Flags","Foodtruck","Genbase","GpositiveGO","HumanGO","Image","Langlog",
//			    "Medical","PlantGO","Scene","Chemistry","Chess","Coffee","Philosophy","VirusGO","Water_quality","Yeast","Yelp"};
//		String[] filename = {"Langlog"};
		String[] filename = {"Philosophy","Corel5k","Tmc2007_500","20NG","HumanGO","Mediamill"};
		for(int testTimes = 0; testTimes<1; testTimes++) {
			for(int fileIndex = 0; fileIndex < 6; fileIndex++) {
				String dataFileName = filename[fileIndex];
				
				String path = "E:\\multiLabel\\DATA\\arff-partitioned\\";
				path += dataFileName;
				path += "\\";
				
				String arffFile_train = path + dataFileName + "-train.arff";
				String arffFile_test = path + dataFileName + "-test.arff";
				String xmlFile = path + dataFileName + ".xml";
				MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
				MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
				
				int dimTrain = origData_train.getNumInstances();
				int dimTest = origData_test.getNumInstances();
				System.out.println(dataFileName + ": " + dimTrain + ":" + dimTest + ":" + origData_train.getNumLabels());
				
				double[] result;
				EvaluationsMul evaluations = new EvaluationsMul();
				
				result = MulLabelLearner.singleLearner_RAkELo(origData_train, origData_test);
//				System.out.println(result.length);
//				System.out.println(result);
				evaluations.prtConclusion(result, dataFileName, "RAkEL"+testTimes);
				
				
//				result = MulLabelLearner.singleLearner_MLKNN(origData_train, origData_test);
//				evaluations.prtConclusion(result, dataFileName, "MLKNN");
				
//				result = MulLabelLearner.singleLearner_RAkEL(origData_train, origData_test);
//				evaluations.prtConclusion(result, dataFileName, "RAkEL");
				
//				result = MulLabelLearner.singleLearner_RAkELd(origData_train, origData_test);
//				evaluations.prtConclusion(result, dataFileName, "RAkELd");
				
//				result = MulLabelLearner.singleLearner_AdaBoostMH(origData_train, origData_test);
//				evaluations.prtConclusion(result, dataFileName, "AdaBoostMH");
				
//				result = MulLabelLearner.singleLearner_ECC(origData_train, origData_test);
//				evaluations.prtConclusion(result, dataFileName, "ECC");
			}
		}
	}
}
