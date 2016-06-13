import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;


import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;


/**
 * Created by dongnanzhy on 1/28/16.
 * This is a class for performing different clustering algorithms.
 */
public class Kmeans {
    private int num_cluster;
    private int num_iterations;
    private double diff;
    private HashMap<Integer, Boolean> isValid;
    private double k;


    public Kmeans () {
        this.num_cluster = 50;
        this.k = 0.6;
        this.num_iterations = 50;
        this.diff = 0.000001;
        isValid = new HashMap<Integer, Boolean>();
        for (int i = 0; i < this.num_cluster; i++) {
            isValid.put(i, true);
        }
    }

    public Kmeans (int num_cluster, int num_iterations, double diff, double k) {
        this.num_cluster = num_cluster;
        this.num_iterations = num_iterations;
        this.diff = diff;
        this.k = k;
        isValid = new HashMap<Integer, Boolean>();
        for (int i = 0; i < num_cluster; i++) {
            isValid.put(i, true);
        }
    }
    /*
       * This function is for users to perform clustering algorithm
     */
    public void findCluster (boolean isDev, boolean isKmeansPlus, boolean isCosine, boolean isIDF, boolean isIDOCLen) throws IOException {
        Matrix d = matrixIO.read(isDev);
        Matrix docs = preProcess(d, isDev, isIDF, isIDOCLen);
        Matrix initCluster = isKmeansPlus? init_plus(docs, isCosine) : init(docs);
        Vector labels = calculate(docs, initCluster, isCosine);
        //Vector labels = assignLabel(docs, cluster, isCosine);
        matrixIO.write(labels, isDev);
    }
    /*
       * Preprocess matrix for feature design, using custom methods
       * Methods including idf weight and document length weight
     */
    private Matrix preProcess (Matrix docs, boolean isDev, boolean isIDF, boolean isIDOCLen) throws IOException {
        if (isIDF && isIDOCLen) {
            HashMap<Integer, Integer> df = matrixIO.readDF(isDev);
            double avgDocLen = docs.zSum() / docs.rowSize();
            for (int i = 0; i < docs.rowSize(); i++) {
                double docLen = docs.viewRow(i).zSum();
                for (int j = 0; j < docs.columnSize(); j++) {
                    double tf = docs.get(i, j);
                    if (tf == 0) continue;
                    //double val = tf / (tf + k1 * ((1-b) + b*docLen/avgDocLen) );
                    double idfWeight = Math.max(0, Math.log((double) (docs.rowSize() - df.get(j) + 0.5) / (df.get(j) + 0.5)));
                    double docLenWeight = 1.0 / (k* docLen/avgDocLen);
                    double val = tf * idfWeight * docLenWeight;
                    docs.set(i, j, val);
                }
            }
        }
        return docs;
    }
    /*
       * K-means algorithm iterations
   */
    private Vector calculate (Matrix docs, Matrix initCluster, boolean isCosine) {
        int iter = 0;
        Matrix curCluster = initCluster;
        Matrix prevCluster = new SparseMatrix(initCluster.rowSize(), initCluster.columnSize());
        prevCluster.assign(0);
        HashMap<Integer,Integer> clusterSize = new HashMap<Integer, Integer>();
        Vector labels = new DenseVector(docs.rowSize());
        while (getDiff(prevCluster, curCluster) > diff && iter < num_iterations) {
            for (int k = 0; k < num_cluster; k++) {
                clusterSize.put(k, 0);
            }
            labels.assign(0);
            // find closest centroids and assign
            for (int i = 0; i < docs.rowSize(); i++) {
                double minScore = Double.MAX_VALUE;
                int minIndex = -1;
                for (int k = 0; k < num_cluster; k++) {
                    // if cluster is null in previous iterations, then remove this center.
                    if (!isValid.get(k)) continue;
                    Vector v1 = docs.viewRow(i);
                    Vector v2 = curCluster.viewRow(k);
                    double score = isCosine? cosineScore(v1, v2) : euclideanScore(v1, v2);
                    if (score < minScore) {
                        minScore = score;
                        minIndex = k;
                    }
                }
                clusterSize.put(minIndex, clusterSize.get(minIndex)+1);
                labels.set(i, minIndex);
            }
            // update previous cluster to the value of current cluster
            prevCluster.assign(curCluster);
            curCluster.assign(0);
            // update centroid of current cluster
            for (int i = 0; i < docs.rowSize(); i++) {
                int index = (int) labels.get(i);
                curCluster.assignRow(index, curCluster.viewRow(index).plus(docs.viewRow(i)));
            }
            for (int k = 0; k < num_cluster; k++) {
               if (clusterSize.get(k) == 0) {
                   isValid.put(k, false);
               } else {
                   curCluster.assignRow(k, curCluster.viewRow(k).divide(clusterSize.get(k)));
               }
            }
            iter++;
            System.out.println("Difference: " + getDiff(prevCluster, curCluster));
        }
        System.out.println("Number of iterations: " + iter);
        System.out.println("Final difference: " + getDiff(prevCluster, curCluster));
        System.out.println("Cluster Size:");
        System.out.println(clusterSize);
        return labels;
    }
    /*
      * K-means cluster initialization, by randomly selecting data points as centroids
    */
    private Matrix init (Matrix docs) {
        int num_docs = docs.rowSize();
        ArrayList<Integer> lst = new ArrayList<Integer>();
        for (int i = 0; i < num_docs; i++) {
            lst.add(i);
        }
        Collections.shuffle(lst);
        Matrix initClusters = new SparseMatrix(num_cluster, docs.columnSize());
        initClusters.assign(0);
        for (int i = 0; i < num_cluster; i++) {
            initClusters.assignRow(i, docs.viewRow(lst.get(i)));
        }
        return initClusters;
    }
    /*
      * K-means++ cluster initialization, by weighted randomly selecting data points as centroids.
    */
    private Matrix init_plus (Matrix docs, boolean isCosine) {
        // Choose one centroid uniformly at random from among the data points.
        int num_docs = docs.rowSize();
        Random rn = new Random();
        int firstDocid = rn.nextInt(num_docs);
        Matrix initClusters = new SparseMatrix(num_cluster, docs.columnSize());
        initClusters.assign(0);
        initClusters.assignRow(0, docs.viewRow(firstDocid));
        // choose following centroids
        Vector weights = new DenseVector(num_docs);
        weights.assign(0);
        for (int k = 1; k < num_cluster; k++) {
            for (int i = 0; i < docs.rowSize(); i++) {
                double minScore = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    Vector v1 = docs.viewRow(i);
                    Vector v2 = initClusters.viewRow(j);
                    double score = isCosine? cosineScore(v1, v2) : euclideanScore(v1, v2);
                    if (score < minScore) {
                        minScore = score;
                    }
                }
//                if (maxScore < 0.00001) {
//                    System.out.println("@@@@@");
//                }
//                double weight = (maxScore == 1) ? 0 : 1/maxScore;
                double weight = isCosine ? (1+minScore) : (minScore*minScore);
                weights.set(i, weight);
            }
            int newCenter = weightRandom(weights);
            initClusters.assignRow(k, docs.viewRow(newCenter));
        }
        return initClusters;
    }
    /*
      * Weighted random selection helper function
    */
    private int weightRandom (Vector weights) {
        double totalWeight = weights.zSum();
        int randomIndex = -1;
        double random = Math.random() * totalWeight;
        for (int i = 0; i < weights.size(); i++)
        {
            random -= weights.get(i);
            if (random <= 0.0d)
            {
                randomIndex = i;
                break;
            }
        }
        return randomIndex;
    }
    /*
      * Assign labels to each document based on cluster matrix.  Not used in this system.
    */
    private Vector assignLabel (Matrix docs, Matrix cluster, boolean isCosine) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        for (int k = 0; k < num_cluster; k++) {
            hm.put(k, 0);
        }
        Vector v = new DenseVector(docs.rowSize());
        for (int i = 0; i < docs.rowSize(); i++) {
            double minScore = Double.MAX_VALUE;
            int minIndex = -1;
            for (int k = 0; k < num_cluster; k++) {
                Vector v1 = docs.viewRow(i);
                Vector v2 = cluster.viewRow(k);
                double score = isCosine? cosineScore(v1, v2) : euclideanScore(v1, v2);
                if (score <= minScore) {
                    minScore = score;
                    minIndex = k;
                }
            }
            v.set(i, minIndex);
            hm.put(minIndex, hm.get(minIndex)+1);
        }
        System.out.println("Cluster Size:");
        System.out.println(hm);
        return v;
    }
    /*
      * Helper function-- calculate differences of two matrices, used for stop criteria of iterations
    */
    private double getDiff (Matrix m1, Matrix m2) {
        //Matrix tmp = m1.minus(m2);
        double diff = 0.0;
        for (int k = 0; k < m1.rowSize(); k++) {
            if (!isValid.get(k)) continue;
            diff += euclideanScore(m1.viewRow(k), m2.viewRow(k));
        }
        return diff;
    }
    /*
      * Helper function-- calculate cosine similarity of two vectors
    */
    private double cosineScore (Vector v1, Vector v2) {
        return -v1.normalize().dot(v2.normalize());
        //return - v1.dot(v2) / (v1.norm(2) * v2.norm(2));
    }
    /*
      * Helper function-- calculate euclidean distance of two vectors
    */
    private double euclideanScore (Vector v1, Vector v2) {
        return v1.minus(v2).norm(2);
    }

    public static void main(String[] args) throws IOException {
        //  Kmeans (int num_cluster, int num_iterations, double diff, double k)
        Kmeans rst =new Kmeans(150,50,0.00001,0.6);
        // findCluster (boolean isDev, boolean isKmeansPlus, boolean isCosine, boolean isIDF, boolean isIDOCLen)
        rst.findCluster(false, true, true, true, true);
    }
}
