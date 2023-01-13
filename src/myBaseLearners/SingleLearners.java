package myBaseLearners;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.data.MultiLabelInstances;
import myStack.Evaluations;
import myStack.Instruments;
import myStack.Logs;
import myStack.Mat;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMO.BinarySMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.ChebyshevDistance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;

public class SingleLearners {
	public static int numKind = 0;
	public static boolean discretizing = true;
	public static String[] allClassifiersName = {};
	
	public static double[] probResult(Classifier classifier, Instances data) throws Exception {
		double[] prob = new double[data.numInstances()];
		for(int i=0; i<prob.length; i++) {
			prob[i] = classifier.distributionForInstance(data.instance(i))[1];
		}
		return prob;
	}
	
	private class CallableTask_train implements Callable<Classifier>{
    	Classifier thisClassifier;
    	Instances trainInstances;
		public CallableTask_train(Classifier mClassifier, Instances mTrain){
			thisClassifier = mClassifier;
			trainInstances = mTrain;
		}
		@Override
		public Classifier call() throws Exception {
			thisClassifier.buildClassifier(trainInstances);
			return thisClassifier;
		}
	}
    //training the base learners in parallel
	public Classifier[] trainThreader(Classifier[] classifiers, Instances[] trains) 
			throws InvalidDataException, ModelInitializationException, Exception {
		int numBase = classifiers.length; 
		Classifier[] trainedClassifiers = new Classifier[numBase];
		ExecutorService service = Executors.newFixedThreadPool(6);
		List<Future<Classifier>> futureList = new ArrayList<Future<Classifier>>();
		
		for (int i = 0; i <numBase; i++) {
			Classifier thisClassifier = classifiers[i];
			Instances thisInstances = trains[i];
			
			CallableTask_train task = new CallableTask_train(thisClassifier, thisInstances);
			futureList.add(service.submit(task));	//in order
		}
		
		for (int j = 0; j < numBase; j++) {
			trainedClassifiers[j] = futureList.get(j).get();
		}
		service.shutdown();
		
		return trainedClassifiers;
	}
	
	
	
	private class CallableTask_test implements Callable<double[]>{
    	Classifier thisClassifier;
    	Instances testInstances;
		public CallableTask_test(Classifier mClassifier, Instances mTest){
			thisClassifier = mClassifier;
			testInstances = mTest;
		}
		@Override
		public double[] call() throws Exception {
			double[] thisprediction = probResult(thisClassifier, testInstances);
			return thisprediction;
		}
	}
    //testing the base learners in parallel
	public double[][] testThreader(Classifier[] classifiers, Instances[] tests) 
			throws InvalidDataException, ModelInitializationException, Exception {
		int numBase = classifiers.length; 
		ExecutorService service = Executors.newFixedThreadPool(6);
		List<Future<double[]>> futureList = new ArrayList<Future<double[]>>();
		double[][] prediction = new double[numBase][];
		
		for (int i = 0; i <numBase; i++) {
			CallableTask_test task = new CallableTask_test(classifiers[i], tests[i]);
			futureList.add(service.submit(task));	//in order
		}
		
		for (int j = 0; j < numBase; j++) {
			prediction[j] = futureList.get(j).get();
		}
		service.shutdown();
		
		return Mat.T(prediction);
	}
	public double testThreader(Classifier[] baseclassifiers, Instances test) 
			throws InvalidDataException, ModelInitializationException, Exception {
		int numBase = baseclassifiers.length; 
		ExecutorService service = Executors.newFixedThreadPool(6);
		List<Future<double[]>> futureList = new ArrayList<Future<double[]>>();
		
		double prediction = 0;
		for (int i = 0; i <numBase; i++) {
			CallableTask_test task = new CallableTask_test(baseclassifiers[i], test);
			futureList.add(service.submit(task));	//in order
		}
		
		for (int j = 0; j < numBase; j++) {
			prediction += futureList.get(j).get()[0];
		}
		service.shutdown();
		return prediction/numBase;
	}
	
	/*
	 * get various of classifiers
	 * */
	public static Classifier getClassifier(int s) {
		switch (s%numKind) {
		case 0:
			return getLogistic();
		case 1:
			return getJ48();
		case 2:
			if(discretizing) {
				return getNaiveBayes();
			}else {
				return getBayesNet();	//Warning: discretizing data set
			}
		case 3:
			return getSMO();
		default:
			System.out.println("ERROR! NO SUCH CLASSIFIER!");
			return null;
		}
	}
	
//	public static Classifier getClassifier(int s) {
//		switch (s%numKind) {
//		case 0:
//			return getLogistic();
//		case 1:
//			return getJ48();
//		case 2:
//			return getBayesNet();	//Warning: discretizing data set
//		case 3:
//			return getRandomTree();
//		case 4:
//			return getJRip();
//		case 5:
//			return getDecisionTable();	//numeric
//		case 6:
//			return getSMO();
//		case 7:
//			return getNaiveBayes();
//		case 8:
//			return getMultilayerPerceptron();	//numeric
//		case 9:
//			return getKNN(5);
//		case 10:
//			return getKNN();
//		default:
//			System.out.println("ERROR! NO SUCH CLASSIFIER!");
//			return null;
//		}
//	}
	
	public static SMO getSMO() {
		SMO smo = new SMO();
//		smo.setKernel(new PolyKernel());
		smo.setKernel(new RBFKernel());
		
		return smo;
	}
	
	public static BinarySMO get_bSMO() {
		SMO smo = new SMO();
		BinarySMO bsmo = smo.new BinarySMO();
		bsmo.setKernel(new PolyKernel());
		return bsmo;
	}
	
	public static IBk getKNN() {
		IBk ibk = new IBk();
		return ibk;
	}
	
	public static IBk getKNN(int k) {
		IBk ibk = new IBk(k);
		return ibk;
	}
	
	public static DistanceFunction getDistFunction(int distFlag) {
		DistanceFunction dFunction = null;
		if(distFlag <= 4) {
			dFunction = getBasisDistFunction(distFlag);
		}
		else {
		}
		return dFunction;
	}
	
	public static DistanceFunction getBasisDistFunction(int distFlag) {
		DistanceFunction dFunction;
		switch (distFlag) {
		case 1:
			dFunction = new ManhattanDistance();
			break;
		case 2:
			dFunction = new EuclideanDistance();
			break;
		case 3:
			dFunction = new MinkowskiDistance();
			break;
		case 4:
			dFunction = new ChebyshevDistance();
			break;
		default:
			System.out.println("SingleLearners ERROR! No such distance function!");
			dFunction = null;
			break;
		}
		return dFunction;
	}
	
	// ALL candidate single-label classification algorithms
	public static IBk getKNN(int k, int metricFlag) throws Exception {
		IBk ibk = new IBk(k);
		DistanceFunction dFunction = getDistFunction(metricFlag);
		LinearNNSearch lnn = new LinearNNSearch();
		lnn.setDistanceFunction(dFunction);
		ibk.setNearestNeighbourSearchAlgorithm(lnn);
		return ibk;
	}
	
	public static J48 getJ48() {
		J48 j48tree = new J48();
		return j48tree;
	}
	
	public static Logistic getLogistic() {
		Logistic logistic = new Logistic();
		return logistic;
	}
	
	public static RandomForest getRandomForest() {
		RandomForest rf = new RandomForest();
		rf.setNumTrees(10);
		return rf;
	}
	
	public static RandomForest getRandomForest(int iter) {
		RandomForest rf = new RandomForest();
		rf.setNumTrees(iter);
		return rf;
	}
	
	public static MultilayerPerceptron getMultilayerPerceptron() {
		MultilayerPerceptron mp = new MultilayerPerceptron();
		return mp;
	}
	
	public static MultilayerPerceptron getMultilayerPerceptron(int epochs, int numHiddens) {
		MultilayerPerceptron mp = new MultilayerPerceptron();
		mp.setTrainingTime(epochs);
		mp.setHiddenLayers(""+numHiddens);
		return mp;
	}
	
	public static DecisionTable getDecisionTable() {
		DecisionTable dt = new DecisionTable();
		return dt;
	}
	
	public static NaiveBayes getNaiveBayes() {
		NaiveBayes nb = new NaiveBayes();
		return nb;
	}
	
	public static BayesNet getBayesNet() {
		BayesNet by = new BayesNet();
		return by;
	}
	
	public static JRip getJRip() {
		JRip jr = new JRip();
		return jr;
	}
	
	public static RandomTree getRandomTree() {
		RandomTree rt = new RandomTree();
		return rt;
	}
	
	public static void main(String[] args) throws Exception {
		numKind = 5;
		System.out.println("Single Test begin.");
//		Instances[] datas = getInstances_MUL("EukaryotePseAAC", 16);
		Instances[] datas = getInstances_MUL("CHD_49", 4);
		Instances traindata = datas[0];
		Instances testdata = datas[1];
		
		for(int i=0; i<5; i++) {
			long t0 = System.currentTimeMillis();
			Classifier mClassifier = getClassifier(i);
			mClassifier.buildClassifier(traindata);
			double[] result = probResult(mClassifier, testdata);
			Logs.logMatrix(result, "======================="+i);
			System.out.println(new Evaluations().evaluateSingle_Acc(result, Instruments.getTarget(testdata)) + "\t" + (System.currentTimeMillis() - t0));
			
			System.out.println(mClassifier.getClass().getName());
		}
		
		System.out.println("ALL FINISHED!");
		System.out.println(Instruments.showMemoryInfo());
	}
	
	public static Instances getInstances() throws Exception {
		Instances data = DataSource.read("D:\\Weka-3-8-4\\data\\diabetes.arff");
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static Instances[] getInstances_MUL() throws Exception {
		String dataFileName = "Birds";
		int label = 0;
		return getInstances_MUL(dataFileName, label);
	}
	
	public static Instances[] getInstances_MUL(int label) throws Exception {
		String dataFileName = "Birds";
		return getInstances_MUL(dataFileName, label);
	}
	
	public static Instances[] getInstances_MUL(String dataFileName, int label) throws Exception {
		String path = "E:\\multiLabel\\code_cui\\mulan-cuiwei\\mulan\\datatest\\ALLdata\\DATA2GEN\\";
		path += dataFileName;
		path += "\\";
		
		String arffFile_train = path + dataFileName + "-train.arff";
		String arffFile_test = path + dataFileName + "-test.arff";
		String xmlFile = path + dataFileName + ".xml";
		MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
		MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
		
//		Logs.logMatrix(Instruments.getTarget(origData_test), "KNN");
		
		Instances singleTrain = Instruments.multi2single(origData_train)[label];
		Instances singleTest = Instruments.multi2single(origData_test)[label];
		
		Instances[] a = {singleTrain, singleTest};
		return a;
	}
}
