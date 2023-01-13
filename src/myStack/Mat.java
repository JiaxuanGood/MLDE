package myStack;

public class Mat {
	public static double[][] getwant(double[][] a, int[] idx){
		int numwant = idx.length;
		double[][] b = new double[numwant][];
		for (int i=0; i<numwant; i++) {
			b[i] = a[i];
		}
		return b;
	}
	public static int[][] getwant(int[][] a, int[] idx){
		int numwant = idx.length;
		int[][] b = new int[numwant][];
		for (int i=0; i<numwant; i++) {
			b[i] = a[i];
		}
		return b;
	}
	
	
	public static double[][] cat(double[][] a, double[][] b){
		if(a==null) {
			return b;
		}
		if(a[0].length!=b[0].length) {
			System.out.println("MAT ERROR: cat.");
			return null;
		}
		double[][] c = new double[a.length+b.length][];
		for(int i=0; i<a.length; i++) {
			c[i] = a[i];
		}
		for(int i=0; i<b.length; i++) {
			c[i+a.length] = b[i];
		}
		return c;
	}
	
	public static double[] init(int dim, double a){
		double[] r = new double[dim];
		for(int i=0; i<dim; i++) {
			r[i] = a;
		}
		return r;
	}
	
	public static double[][] init(int dim, int num, double a){
		double[][] r = new double[dim][num];
		for(int i=0; i<dim; i++) {
			for(int j=0; j<num; j++) {
				r[i][j] = a;
			}
		}
		return r;
	}
	
	public static void init(double[][] mat, double a){
		for(int i=0; i<mat.length; i++) {
			for(int j=0; j<mat[0].length; j++) {
				mat[i][j] = a;
			}
		}
	}
	
	public static void equal(double[][] a, double[][] b){
		if(a.length!=b.length || a[0].length!=b[0].length) {
			System.out.println("isequal:\terror!");
		}
		int cnt = 0;
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[0].length; j++) {
				if(a[i]==b[i]) {
					cnt++;
				}
			}
		}
		if(cnt==a.length*a[0].length) {
			System.out.println("isequal:\ttrue!");
		}
		else {
			System.out.println("isequal:\twrong!");
			System.out.println(Mat.sum_squares(Mat.add(a, Mat.dot(b, -1))));
//			Logs.logMatrix(a, "a");
		}
	}
	
	public static void equal(int[] a, int[] b){
		int len = a.length;
		int rlen = a[len-1];
		if(len!=b.length || rlen!=b[len-1]) {
			System.out.println("isequal:\terror!");
		}
		int[] a2 = new int[rlen];
		int[] b2 = new int[rlen];
		for(int i=0; i<rlen; i++) {
			a2[i] = a[i];
			b2[i] = b[i];
		}
		int tmp;
		int tmp2;
		for(int i=0; i<rlen; i++) {
			for(int j=i+1; j<rlen; j++) {
				if(a2[i]<a2[j]) {
					tmp = a2[i];
					a2[i] = a2[j];
					a2[j] = tmp;
				}
				if(b2[i]<b2[j]) {
					tmp2 = b2[i];
					b2[i] = b2[j];
					b2[j] = tmp2;
				}
			}
		}
		int cnt = 0;
		for(int i=0; i<rlen; i++) {
			if(a2[i]==b2[i]) {
				cnt++;
			}
		}
		if(cnt==rlen) {
			System.out.println("isequal:\ttrue!");
		}
		else {
			System.out.println("isequal:\twrong!");
			Logs.logMatrix_pan(a2);
			Logs.logMatrix_pan(b2);
		}
	}
	
	public static double[][] add(double[][] a, double[][] b){
		if(a.length!=b.length || a[0].length!=b[0].length) {
			System.out.println("MAT ERROR: sub.");
			return null;
		}
		double[][] r = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int k=0; k<a[0].length; k++) {
				r[i][k] = a[i][k] + b[i][k];
			}
		}
		return r;
	}
	
	public static double[] add(double[] a, double[] b){
		if(a.length!=b.length) {
			System.out.println("MAT ERROR: add.");
			return null;
		}
		double[] r = new double[a.length];
		for(int i=0; i<a.length; i++) {
			r[i] = a[i] + b[i];
		}
		return r;
	}
	
	public static double[][] sub(double[][] a, double[][] b){
		if(a.length!=b.length || a[0].length!=b[0].length) {
			System.out.println("MAT ERROR: sub.");
			return null;
		}
		double[][] r = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int k=0; k<a[0].length; k++) {
				r[i][k] = a[i][k] - b[i][k];
			}
		}
		return r;
	}
	
	public static double[][] sub(double[][] a, double b){
		double[][] r = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int k=0; k<a[0].length; k++) {
				r[i][k] = a[i][k] - b;
			}
		}
		return r;
	}
	
	public static double[] sub(double[] a, double[] b) {
		if(a.length!=b.length) {
			System.out.println("MAT ERROR: sub.");
			return null;
		}
		double[] r = new double[a.length];
		for(int i=0; i<a.length; i++) {
			r[i] = a[i] - b[i];
		}
		return r;
	}
	
	public static double[][] dot(double[][] x, double[][] y){	// (l,m)(m,n) -> (l,n)
		if(x[0].length!=y.length) {
			System.out.println("MAT ERROR: dot1.");
			return null;
		}
		int l=x.length, m=y.length, n=y[0].length;
		double[][] r = matrixInitial(l, n, 0.0);
		for(int a=0; a<l; a++) {
			for(int b=0; b<n; b++) {
				for(int c=0; c<m; c++) {
					r[a][b] += x[a][c] * y[c][b];
				}
			}
		}
		return r;
	}
	
	public static double[] dot(double[][] x, double[] y){	// (l,m)(m,1) -> (l,1)
		if(x[0].length!=y.length) {
			System.out.println("MAT ERROR: dot2.");
			return null;
		}
		int l=x.length, m=y.length;
		double[] r = new double[l];
		for(int i=0; i<l; i++) {
			for(int j=0; j<m; j++) {
				r[i] += x[i][j] * y[j];
			}
		}
		return r;
	}
	
	public static double dot(double[] x, double[] y){	// (l,m)(m,1) -> (l,1)
		if(x.length!=y.length) {
			System.out.println("MAT ERROR: dot3.");
			return Double.MIN_VALUE;
		}
		double r = 0;
		for(int i=0; i<x.length; i++) {
			r += x[i] * y[i];
		}
		return r;
	}
	
	public static double[] dot(double[] x, double y){
		int m=x.length;
		double[] r = new double[m];
		for(int a=0; a<m; a++) {
			r[a] = x[a] * y;
		}
		return r;
	}
	
	public static double[][] dot(double[][] x, double y){
		int m=x.length, n=x[0].length;
		double[][] r = new double[m][n];
		for(int a=0; a<m; a++) {
			for(int b=0; b<n; b++) {
				r[a][b] = x[a][b] * y;
			}
		}
		return r;
	}
	
	public static double[][] T(double[][] x){
		double[][] r = new double[x[0].length][x.length];
		for(int a=0; a<x.length; a++) {
			for(int b=0; b<x[0].length; b++) {
//				System.out.println(a + "\t" + b);
				r[b][a] = x[a][b];
			}
		}
		return r;
	}
	
	public static int[][] T(int[][] x){
		int[][] r = new int[x[0].length][x.length];
		for(int a=0; a<x.length; a++) {
			for(int b=0; b<x[0].length; b++) {
				r[b][a] = x[a][b];
			}
		}
		return r;
	}
	
	public static double[][] soft_thresold(double[][] x, double thresold){
		double[][] a = sub(x, thresold);
		double[][] b = sub(dot(x, -1), thresold);
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				if(a[i][j] < 0) {
					a[i][j] = 0;
				}
				if(b[i][j] < 0) {
					b[i][j] = 0;
				}
			}
		}
		return Mat.sub(a, b);
	}
	
	public static double[][] getLap(double[][] Y){	//Y (dim,num)
		double[][] R = getCorrelationes(T(Y));
		double[][] EE = init(R.length, R.length, 1.0);
		return Mat.sub(EE, R);
	}
	
	private static double[][] getCorrelationes(double[][] ab) {
		double[][] correlation = new double[ab.length][ab.length];
		for(int i=0; i<ab.length; i++) {
			correlation[i][i] = 1;
			for(int j=i+1; j<ab.length; j++) {
				correlation[i][j] = getCorrelation(ab[i], ab[j]);
				correlation[j][i] = correlation[i][j];
			}
		}
		return correlation;
	}
//	private static double getCorrelation(double[] a, double[] b) {
//		int dim = a.length;
//		double a_bar = MatrixHandle.sum(a)/dim;
//		double b_bar = MatrixHandle.sum(b)/dim;
//		double correlation = 0;
//		double eaeb=0, da=0, db=0;
//		for(int i=0; i<dim; i++) {
//			eaeb += (a[i]-a_bar)*(b[i]-b_bar);
//			da += Math.pow((a[i]-a_bar), 2);
//			db += Math.pow((b[i]-b_bar), 2);
//		}
//		correlation = eaeb/Math.sqrt(da*db);
//		return correlation;
//	}
	private static double getCorrelation(double[] a, double[] b) {
		double correlation = Mat.dot(a, b) / Math.sqrt( Mat.sum_squares(a)*Mat.sum_squares(b) );
		return correlation;
	}
	
	public static double norm2(double[][] mat) {
		return Math.sqrt(sum_squares(mat));
	}
	public static double sum_squares(double[][] mat) {
		double sum = 0;
		for(int i=0; i<mat.length; i++) {
			for(int k=0; k<mat[0].length; k++) {
				sum += Math.pow(mat[i][k], 2);
			}
		}
		return sum;
	}
	
	public static double sum_squares(double[] mat) {
		double sum = 0;
		for(int i=0; i<mat.length; i++) {
			sum += Math.pow(mat[i], 2);
		}
		return sum;
	}
	
	public static double sum_abs(double[] mat) {
		double sum = 0;
		for(int i=0; i<mat.length; i++) {
			sum += Math.abs(mat[i]);
		}
		return sum;
	}
	
	public static double trace(double[][] a) {
		if(a.length != a[0].length) {
			System.out.println("MAT ERROR: trace.");
			return Double.MIN_VALUE;
		}
		double sum = 0;
		for(int i=0; i<a.length; i++) {
			sum += a[i][i];
		}
		return sum;
	}
	
	public static double[][] INV(double[][] data) {
		// 先是求出行列式的模|data|
		double A = getHL(data);
		// 创建一个等容量的逆矩阵
		double[][] newData = new double[data.length][data.length];
		
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data.length; j++) {
				double num;
				if ((i + j) % 2 == 0) {
					num = getHL(getDY(data, i + 1, j + 1));
				} else {
					num = -getHL(getDY(data, i + 1, j + 1));
				}

				newData[i][j] = num / A;
			}
		}

		// 转置 代数余子式转制
		newData = T(newData);

		return newData;
	}
	
	private static double getHL(double[][] data) {
		// 终止条件
		if (data.length == 2) {
			return data[0][0] * data[1][1] - data[0][1] * data[1][0];
		}

		double total = 0;
		// 根据data 得到行列式的行数和列数
		int num = data.length;
		// 创建一个大小为num 的数组存放对应的展开行中元素求的的值
		double[] nums = new double[num];

		for (int i = 0; i < num; i++) {
			if (i % 2 == 0) {
				nums[i] = data[0][i] * getHL(getDY(data, 1, i + 1));
			} else {
				nums[i] = -data[0][i] * getHL(getDY(data, 1, i + 1));
			}
		}
		for (int i = 0; i < num; i++) {
			total += nums[i];
		}
//		System.out.println("total=" + total);
		return total;
	}
	
	private static double[][] getDY(double[][] data, int h, int v) {
		int H = data.length;
		int V = data[0].length;
		double[][] newData = new double[H - 1][V - 1];

		for (int i = 0; i < newData.length; i++) {

			if (i < h - 1) {
				for (int j = 0; j < newData[i].length; j++) {
					if (j < v - 1) {
						newData[i][j] = data[i][j];
					} else {
						newData[i][j] = data[i][j + 1];
					}
				}
			} else {
				for (int j = 0; j < newData[i].length; j++) {
					if (j < v - 1) {
						newData[i][j] = data[i + 1][j];
					} else {
						newData[i][j] = data[i + 1][j + 1];
					}
				}

			}
		}
		return newData;
	}
	
	public static double[][] dup(double[][] x){
		double[][] y = new double[x.length][x[0].length];
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				y[i][j] = x[i][j];
			}
		}
		return y;
	}
	
	public static double[][] int2double(int[][] a){
		double[][] b = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[0].length; j++) {
				b[i][j] = a[i][j];
			}
		}
		return b;
	}
	
	public static double[] int2double(int[] a){
		double[] b = new double[a.length];
		for(int i=0; i<a.length; i++) {
			b[i] = a[i];
		}
		return b;
	}
	
	public static double[][] eye(int a){
		double[][] b = new double[a][a];
		for(int i=0; i<a; i++) {
			b[i][i] = 1.0;
		}
		return b;
	}
	
	public static double[][] tri(double[][] a, double[][] b, double[][] c){
		int dim = a.length;
		int num = a[0].length;
		double[][] r = new double[dim][num*3];
		for(int i=0; i<dim; i++) {
			for(int j=0; j<num; j++) {
				r[i][j] = a[i][j];
			}
			for(int j=0; j<num; j++) {
				r[i][j+num] = b[i][j];
			}
			for(int j=0; j<num; j++) {
				r[i][j+num*2] = c[i][j];
			}
		}
		return r;
	}
	
	public static double[][] matrixDup2(int[][] x){
		double[][] y = new double[x.length][x[0].length];
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				y[i][j] = x[i][j];
			}
		}
		return y;
	}
	
	public static double[][] matrixDup2(double[][] x){
		double[][] y = new double[x.length][x[0].length];
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				y[i][j] = x[i][j];
			}
		}
		return y;
	}

	public static double[][] matrixInitial(int dim, int num, double a){
		double[][] r = new double[dim][num];
		for(int i=0; i<dim; i++) {
			for(int j=0; j<num; j++) {
				r[i][j] = a;
			}
		}
		return r;
	}
	
	public static double[] matrixInitial(int dim, double a){
		double[] r = new double[dim];
		for(int i=0; i<dim; i++) {
			r[i] = a;
		}
		return r;
	}
	
	public static int[][] matrixInitial(int dim, int num, int a){
		int[][] r = new int[dim][num];
		for(int i=0; i<dim; i++) {
			for(int j=0; j<num; j++) {
				r[i][j] = a;
			}
		}
		return r;
	}
	
	public static double[] multiplic(double[] x, double y){	//(l,m)(m,n)
		int m=x.length;
		double[] r = new double[m];
		for(int a=0; a<m; a++) {
			r[a] = x[a] * y;
		}
		return r;
	}
	
	public static double[][] multiplic(double[][] x, double y){	//(l,m)(m,n)
		int m=x.length, n=x[0].length;
		double[][] r = new double[m][n];
		for(int a=0; a<m; a++) {
			for(int b=0; b<n; b++) {
				r[a][b] = x[a][b] * y;
			}
		}
		return r;
	}
	
	public static double[][] multiplic(double[][] x, double[][] y){	//(l,m)(m,n)
		int l=x.length, m=y.length, n=y[0].length;
		double[][] r = matrixInitial(l, n, 0.0);
		for(int a=0; a<l; a++) {
			for(int b=0; b<n; b++) {
				for(int c=0; c<m; c++) {
					r[a][b] += x[a][c] * y[c][b];
				}
			}
		}
		return r;
	}
	
		public static double sum(int[] mat) {
		double sum = 0;
		for(int i=0; i<mat.length; i++) {
			sum += mat[i];
		}
		return sum;
	}
	
	public static double sum(double[] mat) {
		double sum = 0;
		for(int i=0; i<mat.length; i++) {
			sum += mat[i];
		}
		return sum;
	}
	
		public static double[] vectcat(double[] a, double b) {
		double[] r = new double[a.length+1];
		int i=0;
		for(; i<a.length; i++) {
			r[i] = a[i];
		}
		r[i] = b;
		return r;
	}
	
	public static double[] vectcat(double[] a, double b[]) {
		double[] r = new double[a.length+b.length];
		int i=0;
		for(; i<a.length; i++) {
			r[i] = a[i];
		}
		for(int j=0; j<b.length; i++,j++) {
			r[i] = b[j];
		}
		return r;
	}

}
