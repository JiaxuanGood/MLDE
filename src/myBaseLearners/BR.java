package myBaseLearners;

import java.io.File;

import mulan.data.MultiLabelInstances;
import myStack.EvaluationsMul;
import myStack.Instruments;
import myStack.Logs;
import myStack.Mat;
import myStack.Run_MLDE;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class BR {
	boolean save;
	String data;
	int numLabel = -1;
	int classifierId = -1;
//	public boolean isMultiThread = true;
	Classifier[] mBR;
	
	public BR() {
	}
	
	// train
	public BR(int ClassifierID, MultiLabelInstances origData_train) throws Exception {
		save = false;
		numLabel = origData_train.getNumLabels();
		classifierId = ClassifierID;
		mBR = new Classifier[numLabel];
		train(origData_train);
	}
	
	// train and save
	public BR(int ClassifierID, MultiLabelInstances origData_train, String dataset) throws Exception {
		save = true;
		data = dataset;
		numLabel = origData_train.getNumLabels();
		classifierId = ClassifierID;
		mBR = new Classifier[numLabel];
		String path = "models\\"+data+"\\";
		File mFile = new File(path);
		mFile.mkdir();
		train(origData_train);
		for (int i=0; i<numLabel; i++) {
			SerializationHelper.write(path + ClassifierID+"_"+i+".model", mBR[i]);
		}
	}
	
	// read
	public BR(String dataset, int ClassifierID, int num_label) throws Exception {
		save = true;
		numLabel = num_label;
		classifierId = ClassifierID;
		String path = "models\\"+dataset+"\\";
		mBR = new Classifier[numLabel];
		for (int i=0; i<numLabel; i++) {
			String thispath = path + ClassifierID + "_" + i + ".model";
			mBR[i] = (Classifier) SerializationHelper.read(thispath);
		}
	}
	
	private void train(MultiLabelInstances origData_train) throws Exception {
		Instances[] slData_train = Instruments.multi2single(origData_train);
		for (int i=0; i<numLabel; i++) {
			mBR[i] = SingleLearners.getClassifier(classifierId);
		}
		if(Run_MLDE.isMultiThread) {
			SingleLearners sL = new SingleLearners();
			mBR = sL.trainThreader(mBR, slData_train);
		}else {
			for (int i=0; i<numLabel; i++) {
				mBR[i].buildClassifier(slData_train[i]);
			}
		}
	}
	
	public static boolean isExist(String dataset) {
		String path = "models\\"+dataset+"\\";
		File mFile = new File(path);
		return mFile.exists();
	}
	
	public double[][] predict(MultiLabelInstances origData_test) throws Exception {
		if(Run_MLDE.isMultiThread) {
			return predict_mulThread(origData_test);
		}else {
			return predict_prob(origData_test);
		}
	}
	
	private double[][] predict_mulThread(MultiLabelInstances origData_test) throws Exception {
		Instances[] slData_test = Instruments.multi2single(origData_test);
		SingleLearners sL = new SingleLearners();
		double[][] prediction = sL.testThreader(mBR, slData_test);
		return prediction;
	}
	
	private double[][] predict_prob(MultiLabelInstances origData_test) throws Exception {
		Instances[] slData_test = Instruments.multi2single(origData_test);
		double[][] predict = new double[numLabel][];
		for (int i=0; i<numLabel; i++) {
			predict[i] = SingleLearners.probResult(mBR[i], slData_test[i]);
		}
		return Mat.T(predict);
	}
	
	public double predict_prob_mulThread(BR[] mBRs, MultiLabelInstances origData_test, int label) throws Exception {
		Classifier[] mClassifiers = new Classifier[mBRs.length];
		for(int i=0; i<mBRs.length; i++) {
			mClassifiers[i] = mBRs[i].mBR[label];
		}
		Instances slData_test_a = Instruments.multi2single(origData_test, label);
		SingleLearners sL = new SingleLearners();
		double prediction = sL.testThreader(mClassifiers, slData_test_a);
		return prediction;
	}
	
	public double[] predict_prob(MultiLabelInstances origData_test, int label) throws Exception {
		Instances slData_test_a = Instruments.multi2single(origData_test, label);
		return SingleLearners.probResult(mBR[label], slData_test_a);
	}
	
	public static void main(String[] args) throws Exception {
		String[] filename = {"CHD_49","Emotions","Flags","Water_quality"};	//4 tiny
		for(int fileIndex = 0; fileIndex < 1; fileIndex++) {
			long t0 = System.currentTimeMillis();
			
			String dataFileName = filename[fileIndex];
			dataFileName = "Yeast";
			
			String path = "E:\\multiLabel\\code_cui\\mulan-cuiwei\\mulan\\datatest\\ALLdata\\DATA2GEN\\";
			path += dataFileName;
			path += "\\";
			
			String arffFile_train = path + dataFileName + "-train.arff";
			String arffFile_test = path + dataFileName + "-test.arff";
			String xmlFile = path + dataFileName + ".xml";
			MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
			MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
			
			BR mBR = null;
			if(BR.isExist(dataFileName)) {
				mBR = new BR(dataFileName, 2, origData_train.getNumLabels());
			}
			else {
				mBR = new BR(2, origData_train, dataFileName);
			}
			
			double[][] scores = mBR.predict(origData_test);
			double[][] scores2 = mBR.predict_prob(origData_test);
			int[][] t_test = Instruments.getTarget(origData_test);
			
			EvaluationsMul evaluations = new EvaluationsMul();
			double[] result =  evaluations.evaluating(scores, t_test);
			double[] result2 =  evaluations.evaluating(scores2, t_test);
			evaluations.prtConclusion(result, (System.currentTimeMillis()-t0), dataFileName, "CC_SMO");
			evaluations.prtConclusion(result2, (System.currentTimeMillis()-t0), dataFileName, "CC_SMO");
			
			Logs.logMatrix(scores, "scores");
			Logs.logMatrix(scores2, "scores2");
		}
		
		System.out.println("ALL FINISHED");
	}
}
