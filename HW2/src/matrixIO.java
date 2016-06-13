import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by dongnanzhy on 1/28/16.
 * This is a class for read docVectors or df values, and write final result to specific path
 */
public class matrixIO {
    static final int NUM_DOC_DEV = 942;
    static final int NUM_TERM_DEV = 14063;
    static final int NUM_DOC_TEST = 942;
    static final int NUM_TERM_TEST = 13924;
    static final String DEV_INPUT_PATH = "HW2_data/HW2_dev.docVectors";
    static final String TEST_INPUT_PATH = "HW2_data/HW2_test.docVectors";
    static final String DEV_OUT_PATH = "HW2_data/HW2_dev.clusters";
    static final String TEST_OUT_PATH = "HW2_data/HW2_test.clusters";
    static final String DEV_DF_PATH = "HW2_data/HW2_dev.df";
    static final String TEST_DF_PATH = "HW2_data/HW2_test.df";
    static final String DEV_GOLDSTAND_PATH = "HW2_data/HW2_dev.gold_standards";

    public static Matrix read(boolean isDev) throws IOException {
        File file;
        Matrix m;
        if (isDev) {
            m = new SparseMatrix(NUM_DOC_DEV, NUM_TERM_DEV);
            file = new File(DEV_INPUT_PATH);
        } else {
            m = new SparseMatrix(NUM_DOC_TEST, NUM_TERM_TEST);
            file = new File(TEST_INPUT_PATH);
        }
        m.assign(0);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        int num_docs = 0;
        int num_words = 0;
        int num_uniqWords_doc = 0;
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        try
        {
            String line = null;
            int row = 0;
            while ((line = reader.readLine()) != null)
            {
                String[] index = line.split(" ");
                if (isDev && row == 0) {
                    System.out.println("The total number of unique words in first document: " + index.length);
                }
                num_uniqWords_doc += index.length;
                ArrayList<Integer> wordID = new ArrayList<Integer>();
                for (String s: index) {
                    String[] val = s.split(":");
                    m.set(row, Integer.valueOf(val[0]), Double.valueOf(val[1]));

                    if (isDev && row == 0 && Integer.valueOf(val[1]) == 2) {
                        wordID.add(Integer.valueOf(val[0]));
                    }
                    num_words += Integer.valueOf(val[1]);
                    if (!hm.containsKey(Integer.valueOf(val[0]))) {
                        hm.put(Integer.valueOf(val[0]), 1);
                    }
                }
                if (isDev && row == 0) {
                    System.out.println("All of the word ids: " + wordID);
                }
                row++;
                num_docs++;
            }
        } catch (IOException ex)
        {
            ex.printStackTrace();
        } finally
        {
            reader.close();
            System.out.println("Total number of documents: " + num_docs);
            System.out.println("Total number of words: " + num_words);
            System.out.println("Total number of unique words: " + hm.size());
            System.out.println("Average number of unique words per document: " + (double) num_uniqWords_doc / num_docs);
        }
        return m;
    }


    public static HashMap<Integer, Integer> readDF (boolean isDev) throws IOException {
        File file;
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        if (isDev) {
            file = new File(DEV_DF_PATH);
        } else {
            file = new File(TEST_DF_PATH);
        }
        BufferedReader reader = new BufferedReader(new FileReader(file));
        try
        {
            String line = null;
            while ((line = reader.readLine()) != null)
            {
                String[] dfVal = line.split(":");
                hm.put(Integer.valueOf(dfVal[0]), Integer.valueOf(dfVal[1]));
            }
        } catch (IOException ex)
        {
            ex.printStackTrace();
        } finally
        {
            reader.close();
        }
        return hm;
    }

    public static void readGoldStand() throws IOException {
        File file = new File(DEV_GOLDSTAND_PATH);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        HashMap<String, Integer> hm = new HashMap<String, Integer>();
        try {
            String line = null;
            while ((line = reader.readLine()) != null) {
                if (!hm.containsKey(line)) {
                    hm.put(line, 1);
                }
            }
        } catch (IOException ex)
        {
            ex.printStackTrace();
        } finally
        {
            reader.close();
            System.out.println("Total number of clusters: " + hm.size());
        }
    }

    public static void write (Vector result, boolean isDev) throws IOException {
        File file;
        if (isDev) {
            file = new File(DEV_OUT_PATH);
        } else {
            file = new File(TEST_OUT_PATH);
        }
        FileWriter writer = new FileWriter(file);
        for (int i = 0; i < result.size(); i++) {
            writer.write(i + " " + (int) (result.get(i)) + "\n");
        }
        writer.close();
    }

}
