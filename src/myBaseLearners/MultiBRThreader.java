package myBaseLearners;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import mulan.data.MultiLabelInstances;

public class MultiBRThreader {
	private class CallableTask_train implements Callable<BR>{
		MultiLabelInstances trainInstances;
		int classifierID;
		public CallableTask_train(MultiLabelInstances mTrain, int index){
			trainInstances = mTrain;
			classifierID = index;
		}
		@Override
		public BR call() throws Exception {
			BR thisClassifier = new BR(classifierID, trainInstances);
			return thisClassifier;
		}
	}
	
	public BR[] trainThreader(MultiLabelInstances[] trains) throws InterruptedException, ExecutionException  {
		int numBase = trains.length; 
		BR[] trainedClassifiers = new BR[numBase];
		ExecutorService service = Executors.newFixedThreadPool(6);
		List<Future<BR>> futureList = new ArrayList<Future<BR>>();
		
		for (int i = 0; i <numBase; i++) {
			MultiLabelInstances thisInstances = trains[i];
			CallableTask_train task = new CallableTask_train(thisInstances, i);
			futureList.add(service.submit(task));	//in order
		}
		
		for (int j = 0; j < numBase; j++) {
			trainedClassifiers[j] = futureList.get(j).get();
		}
		service.shutdown();
		
		return trainedClassifiers;
	}
}
