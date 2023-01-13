package myStack;

/*
 * author JiaxuanLi
 * */

public class Evaluations {
	private double threshold = 0.5;
	
	public Evaluations() {
	}
	
	public Evaluations(double diythreshold) {
		threshold = diythreshold;
	}
	
	private int[] double2int(double[] a){
		int[] r = new int[a.length];
		for(int i=0; i<a.length; i++){
			if(a[i]>=threshold){
				r[i] = 1;
			}
			else{
				r[i] = 0;
			}
		}
		return r;
	}
	
	public double evaluateSingle_Acc(double[] output, int[] target) {
		if(output.length != target.length) {
			return Double.MIN_VALUE;
		}
		int[] y = double2int(output);
		return evaluateAcc(y, target);
	}
	
	public double[] evaluateSingle_Accs(double[][] output, int[][] target) {
		output = Mat.T(output);
		target = Mat.T(target);
		int numLabel = output.length;
		double[] accs = new double[numLabel];
		for(int i=0; i<numLabel; i++) {
			accs[i] = evaluateSingle_Acc(output[i], target[i]);
		}
		return accs;
	}
	
	public double evaluateAcc(int[] output, int[] target) {
		if(output.length != target.length) {
			return Double.MIN_VALUE;
		}
		int cnt = 0;
		for(int i=0; i<target.length; i++) {
			if(output[i] == target[i]) {
				cnt++;
			}
		}
		return (double)cnt/target.length;
	}
	
	public double[] evaluate_metrics(int[] abcd) {
		int a = abcd[0];
		int b = abcd[1];
		int c = abcd[2];
		int d = abcd[3];
		double acc = (double)(a+d)/(a+b+c+d);
		double pre = (double)a/(a+c);
		if(!(pre<=1)) {
			pre=0;
		}
		double rec = (double)a/(a+b);
		if(!(rec<=1)) {
			rec=0;
		}
		double f1 = 2*pre*rec/(pre+rec);
		if(!(f1<=1)) {
			f1=0;
		}
		double[] result = {acc, pre, rec, f1};
		return result;
	}
	
}
