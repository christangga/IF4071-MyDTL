package newdtl;

import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;

public class NewDTL {

    public static void main(String[] args) {
        Scanner scn = new Scanner(System.in);
        System.out.print("Masukkan tipe algoritma (id3, new1d3, j48, newj48, svm): ");
        String input = scn.next();
        while (!input.equals("exit")) {
            System.out.print("Masukkan data set: ");
            String source = scn.next();
            String dataSource = "data/" + source + ".arff";
            String dataSourceUnlabeled = "data/" + source + ".unlabeled.arff";

            // load file from data source
            Instances instances = Helper.loadDataFromFile(dataSource);

            switch (input) {
                case "id3":
                    Classifier id3 = Helper.buildClassifier(instances, "id3");
                    System.out.println(id3.toString() + "\n");
                    Helper.classifyUsingModel(id3, dataSourceUnlabeled);
                    Helper.tenFoldCrossValidation(instances, id3);
                    break;
                case "newid3":
                    Classifier newId3 = Helper.buildClassifier(instances, "newid3");
                    System.out.println(newId3.toString() + "\n");
                    Helper.classifyUsingModel(newId3, dataSourceUnlabeled);
                    Helper.tenFoldCrossValidation(instances, newId3);
                    break;
                case "j48":
                    Classifier j48 = Helper.buildClassifier(instances, "j48");
                    System.out.println(j48.toString() + "\n");
                    Helper.classifyUsingModel(j48, dataSourceUnlabeled);
                    Helper.tenFoldCrossValidation(instances, j48);
                    break;
                case "newj48":
                    Classifier newJ48 = Helper.buildClassifier(instances, "newj48");
                    System.out.println(newJ48.toString() + "\n");
                    Helper.classifyUsingModel(newJ48, dataSourceUnlabeled);
                    Helper.tenFoldCrossValidation(instances, newJ48);
                    break;
                case "svm":
                    Classifier svm = Helper.buildClassifier(instances, "svm");
                    System.out.println(svm.toString() + "\n");
                    Helper.classifyUsingModel(svm, dataSourceUnlabeled);
                    Helper.tenFoldCrossValidation(instances, svm);
                    break;
                default:
                    break;
            }

            System.out.println("=================================================");
            System.out.print("Masukkan tipe algoritma (id3, new1d3, j48, newj48, svm): ");
            input = scn.next();
        }
    }
}
