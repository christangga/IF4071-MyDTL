package mydtl;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class Helper {

    public Helper() {
    }

    public static Instances loadDataFromFile(String file) {
        Instances data = null;

        try {
            data = DataSource.read(file);

            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return data;
    }

    /* 
     * attribute - a string representing the list of attributes.
     * Since the string will typically come from a user, attributes are indexed from 1. 
     * eg: first-3,5,6-last
     */
    public static Instances removeAttribute(Instances data, String attribute) {
        Instances newData = null;

        try {
            Remove remove = new Remove();
            remove.setAttributeIndices(attribute);
            remove.setInputFormat(data);
            newData = Filter.useFilter(data, remove);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return newData;
    }

    public static Instances resample(Instances data) {
        Instances newData = null;
        
        try {
            Resample resample = new Resample();
            resample.setInputFormat(data);
            newData = Filter.useFilter(data, resample);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return newData;
    }
    
    public static Classifier buildClassifier(Instances data, String type) {
        try {
            switch (type.toLowerCase()) {
                case "naivebayes":
                    NaiveBayes naiveBayes = new NaiveBayes();
                    naiveBayes.buildClassifier(data);
                    
                    return naiveBayes;
                case "id3":
                    Id3 id3 = new Id3();
                    id3.buildClassifier(data);
                    
                    return id3;
                case "j48":
                    J48 j48 = new J48();
                    j48.buildClassifier(data);
                    
                    return j48;
            }
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return null;
    }
    
    public static void tenFoldCrossValidation(Instances data,
        Classifier classifier) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data,
                10, new Random(1));
            System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static void saveModelToFile(Instances data,
        Classifier classifier, String file) {
        try {
            Classifier cls = classifier;
            cls.buildClassifier(data);

            ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(file));
            oos.writeObject(cls);

            oos.flush();
            oos.close();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static Classifier loadModelFromFile(String file) {
        Classifier cls = null;

        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
                file));
            cls = (Classifier) ois.readObject();

            ois.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return cls;
    }

    public static void classifyUsingModel(Instances data,
        Classifier classifier) {
        try {
            Classifier cls = classifier;
            cls.buildClassifier(data);

            Instances unlabeled = DataSource
                .read("weather.nominal_unlabeled.arff");
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

            Instances labeled = new Instances(unlabeled);
            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = cls.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.toString());
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
