package mydtl;

import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

/**
 *
 * @author susanti_2
 */
public class MyJ48 extends Classifier {

    private final double MISSING_VALUE = Double.NaN;
    private final double DOUBLE_COMPARE_VALUE = 1e-6;

    /**
     * The node's children.
     */
    private MyJ48[] m_Children;

    /**
     * Attribute used for splitting.
     */
    private Attribute m_Attribute;

    /**
     * Class value if node is leaf.
     */
    private double m_Label;
    
    /**
     * Class distribution if node is leaf.
     */
    private double[] m_ClassDistribution;

    /**
     * Class attribute of dataset.
     */
    private Attribute m_ClassAttribute;

    
    private boolean m_IsLeaf;
    
    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds myj48 tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier failed to build
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // Mengecek apakah data dapat dibuat classifier
        getCapabilities().testWithFail(data);

        // Menghapus instances dengan missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makePrunedTree(data);
    }

    private void makePrunedTree(Instances data) throws Exception {
        makeTree(data);
        pruneTree(data);
    }
    
    /**
     * Creates an myj48 tree.
     *
     * @param data the training data
     * @exception Exception if tree failed to build
     */
    private void makeTree(Instances data) throws Exception {

        m_IsLeaf = false;

        // Mengecek apakah tidak terdapat instance yang dalam node ini
        if (data.numInstances() == 0) {
            m_IsLeaf = true;
            m_Attribute = null;
            m_Label = MISSING_VALUE;
            m_ClassDistribution = new double[data.numClasses()];  
        } else {
            // Mencari IG maksimum
            double[] gainRatios = new double[data.numAttributes()];

            data = toNominalInstances(data);

            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                gainRatios[att.index()] = computeGainRatio(data, att);
            }

            m_Attribute = data.attribute(maxIndex(gainRatios));

            // Menghitung distribusi kelas untuk node tersebut
            m_ClassDistribution = new double[data.numClasses()];
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = (Instance) data.instance(i);
                m_ClassDistribution[(int) inst.classValue()]++;
            }
            m_ClassAttribute = data.classAttribute();
            
            // Membuat daun jika IG-nya 0
            if (doubleEqual(gainRatios[m_Attribute.index()], 0)) {
                m_Attribute = null;
                m_IsLeaf = true;

                m_Label = maxIndex(m_ClassDistribution);
                
            } else {
                if (isMissing(data, m_Attribute)) {
                    //cari modus
                    int index = findModus(data, m_Attribute);                    
                    //ubah data yang punya missing value
                    Enumeration dataEnum = data.enumerateInstances();
                    while (dataEnum.hasMoreElements()) {
                        Instance inst = (Instance) dataEnum.nextElement();
                        if (inst.isMissing(m_Attribute)) {
                            inst.setValue(m_Attribute, m_Attribute.value(index));
                        }
                    }
                }
                // Membuat tree baru di bawah node ini
                Instances[] splitData = splitData(data, m_Attribute);
                m_Children = new MyJ48[m_Attribute.numValues()];
                for (int j = 0; j < m_Attribute.numValues(); j++) {
                    m_Children[j] = new MyJ48();
                    m_Children[j].makeTree(splitData[j]);
                }
            }
        }
    }
    
    /**
     * search data that has missing value for attribute
     * @param data the data for searching
     * @param attr the attribute for searching
     * @return if data has missing value for attribute
     */
    private boolean isMissing(Instances data, Attribute attr) {
        boolean isMissing=false;
        Enumeration dataEnum = data.enumerateInstances();
        while (dataEnum.hasMoreElements() && !isMissing) {
            Instance inst = (Instance) dataEnum.nextElement();
            if (inst.isMissing(attr)) {
                isMissing = true;
            }
        }
        return isMissing;
    }
    
    /**
     * search index of attribute that has most common value
     * @param data the data for searching
     * @param attr the attribute for searching
     * @return index of attribute that has most common value
     */
    private int findModus(Instances data, Attribute attr) {
        //cari modus
        int[] modus = new int[attr.numValues()];
        Enumeration dataEnumeration = data.enumerateInstances();

        while (dataEnumeration.hasMoreElements()) {
            Instance inst = (Instance) dataEnumeration.nextElement();
            if (!inst.isMissing(attr)) {
                modus[(int) inst.value(attr)]++;
            }
        }
        //cari modus terbesar
        int max=0;
        int index=-1;
        for(int i=0 ;i< modus.length;++i) {
            if(modus[i] > max) {
                max = modus[i];
                index = i;
            }   
        }
        return index;
    }
    
    /**
     * Use the list of rule to prune the created tree
     */
    private void pruneTree(Instances data) {
        
        if (!m_IsLeaf){

            // Prune all subtrees.
            for (int i=0;i<m_Children.length;i++) {
                System.out.println("==========================sblm");
                System.out.println(this.toString());
                m_Children[i].pruneTree(data);
                System.out.println(this.toString());
                System.out.println("==========================sesudah");
            }
            
            // Cari nilai Backup error dari node tersebut 
            double backupError = backUpError();
            double staticError = staticErrorEstimate((int) DoubleStream.of(m_ClassDistribution).sum(), 
                    (int) m_ClassDistribution[maxIndex(m_ClassDistribution)], m_ClassDistribution.length);
            
            if(backupError > staticError) {
                m_Attribute = null;
                m_Label = maxIndex(m_ClassDistribution);
                m_Children = null;
                m_IsLeaf = true;
            }
        }
    }
    
    /**
     * Convert Instances with numeric attributes to nominal attributes
     *
     * @param data the data to be converted
     * @return Instances with nominal attributes
     */
    private Instances toNominalInstances(Instances data) {
        for (int ix = 0; ix < data.numAttributes(); ++ix) {
            Attribute att = data.attribute(ix);
            if (data.attribute(ix).isNumeric()) {

                // Get an array of integer that consists of distinct values of the attribute
                HashSet<Double> numericSet = new HashSet<>();
                for (int i = 0; i < data.numInstances(); ++i) {
                    numericSet.add(data.instance(i).value(att));
                }

                Double[] numericValues = new Double[numericSet.size()];
                int iterator = 0;
                for (Double i : numericSet) {
                    numericValues[iterator] = i;
                    iterator++;
                }

                // Sort the array
                sortArray(numericValues);

                // Search for threshold and get new Instances
                double[] infoGains = new double[numericValues.length - 1];
                Instances[] tempInstances = new Instances[numericValues.length - 1];
                for (int i = 0; i < numericValues.length - 1; ++i) {
                    tempInstances[i] = convertInstances(data, att, numericValues[i]);
                    try {
                        infoGains[i] = computeGainRatio(tempInstances[i], tempInstances[i].attribute(att.name()));
                    } catch (Exception e) {
                    }
                }

                data = new Instances(tempInstances[maxIndex(infoGains)]);
            }
        }
        return data;
    }

    /**
     * Convert all instances attribute type and values into nominal
     *
     * @param data the data to be converted
     * @param att attribute to be changed to nominal
     * @param threshold the threshold for attribute value
     * @return Instances with all converted values
     */
    private static Instances convertInstances(Instances data, Attribute att, double threshold) {
        Instances newData = new Instances(data);

        // Add attribute		
        try {
            Add filter = new Add();
            filter.setAttributeIndex((att.index() + 2) + "");
            filter.setNominalLabels("<=" + threshold + ",>" + threshold);
            filter.setAttributeName(att.name() + "temp");
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (int i = 0; i < newData.numInstances(); ++i) {
            if ((double) newData.instance(i).value(newData.attribute(att.name())) <= threshold) {
                newData.instance(i).setValue(newData.attribute(att.name() + "temp"), "<=" + threshold);
            } else {
                newData.instance(i).setValue(newData.attribute(att.name() + "temp"), ">" + threshold);
            }
        }

        Instances finalData = Helper.removeAttribute(newData, (att.index() + 1) + "");
        finalData.renameAttribute(finalData.attribute(att.name() + "temp"), att.name());

        return finalData;
    }

    /**
     * Sort an array of integer using bubble sort algorithm
     *
     * @param arr the array to be sorted
     */
    private static void sortArray(Double[] arr) {
        double temp;
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 1; j < arr.length - i; j++) {
                if (arr[j - 1] > arr[j]) {
                    temp = arr[j - 1];
                    arr[j - 1] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    /**
     * Normalize the values in array of double
     *
     * @param array the array of double
     */
    private void normalizeDouble(double[] array) {
        double sum = 0;
        for (double d : array) {
            sum += d;
        }

        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < array.length; ++i) {
                array[i] /= sum;
            }
        } else {
            // Do nothing		
        }
    }

    /**
     * Check whether two double values are the same
     *
     * @param d1 the first double value
     * @param d2 the second double value
     * @return true if the values are the same, false if not
     */
    private boolean doubleEqual(double d1, double d2) {
        return (d1 == d2) || Math.abs(d1 - d2) < DOUBLE_COMPARE_VALUE;
    }

    /**
     * Search for index with largest value from array of double
     *
     * @param array the array of double
     * @return index of array with maximum value
     */
    private static int maxIndex(double[] array) {
        double max = 0;
        int index = 0;

        if (array.length > 0) {
            for (int i = 0; i < array.length; ++i) {
                if (array[i] > max) {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        } else {
            return -1;
        }
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    @Override
    public double classifyInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (m_Attribute == null) {
            return m_Label;
        } else {
            boolean isComparison = false;
            Enumeration enumeration = m_Attribute.enumerateValues();
            String val = null;
            while (enumeration.hasMoreElements()) {
                val = (String) enumeration.nextElement();
                if (val.contains("<")) {
                    isComparison = true;
                    break;
                }
            }

            if (isComparison) {
                double threshold = getThreshold(val);
                double instanceValue = (double) instance.value(m_Attribute);

                if (instanceValue <= threshold) {
                    instance.setValue(m_Attribute, "<=" + String.valueOf(threshold));
                } else {
                    instance.setValue(m_Attribute, ">" + String.valueOf(threshold));
                }
            }
            return m_Children[(int) instance.value(m_Attribute)].
                classifyInstance(instance);
        }
    }

    /**
     * Parse a string of value to get its threshold e.g. "<=24" means the
     * threshold is 24
     *
     * @param val the string to be parsed
     * @return the threshold parsed from the string
     */
    private double getThreshold(String val) {
        return Double.parseDouble(val.replace("<=", ""));
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    @Override
    public double[] distributionForInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyJ48: Cannot handle missing values");
        }
        if (m_Attribute == null) {
            return m_ClassDistribution;
        } else {
            return m_Children[(int) instance.value(m_Attribute)].
                distributionForInstance(instance);
        }
    }

    /**
     * Prints the decision tree using the private toString method from below.
     *
     * @return a textual description of the classifier
     */
    @Override
    public String toString() {

        if ((m_ClassDistribution == null) && (m_Children == null)) {
            return "MyJ48: No model built yet.";
        }
        return "MyJ48\n\n" + toString(0);
    }

    /**
     * Computes Gain Ratio for an attribute.
     *
     * @param data the data for which gain ratio is to be computed
     * @param att the attribute
     * @return the gain ratio for the given attribute and data
     * @throws Exception if computation fails
     */
    private static double computeGainRatio(Instances data, Attribute att)
        throws Exception {

        double infoGain = computeInfoGain(data, att);
        double splitInfo = computeSplitInformation(data, att);
        return infoGain > 0 ? infoGain / splitInfo : infoGain;
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    private static double computeInfoGain(Instances data, Attribute att)
        throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double dataNumInstances = data.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                infoGain -= proportion * computeEntropy(splitdata);
            }
        }
        return infoGain;
    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data class distribution
     * @throws Exception if computation fails
     */
    private static double computeEntropy(Instances data) throws Exception {

        double[] labelCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); ++i) {
            labelCounts[(int) data.instance(i).classValue()]++;
        }

        double entropy = 0;
        for (int i = 0; i < labelCounts.length; i++) {
            if (labelCounts[i] > 0) {
                double proportion = labelCounts[i] / data.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }

    /**
     * Computes Split information for an attribute.
     *
     * @param data the data for which split information is to be computed
     * @param att the attribute
     * @return the split information for the given attribute and data
     * @throws Exception if computation fails
     */
    private static double computeSplitInformation(Instances data, Attribute att) throws Exception {

        double splitInfo = 0;
        Instances[] splitData = splitData(data, att);
        double dataNumInstances = data.numInstances();

        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                splitInfo -= proportion * log2(proportion);
            }
        }
        return splitInfo;
    }

    /**
     * Count the logarithm value with base 2 of a number
     *
     * @param num number that will be counted
     * @return logarithm value with base 2
     */
    private static double log2(double num) {
        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }

    /**
     * split the dataset based on attribute
     *
     * @param data dataset used for splitting
     * @param att attribute used to split the dataset
     * @return
     */
    private static Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }
        return splitData;
    }

    /**
     * Outputs a tree at a certain level.
     *
     * @param level the level at which the tree is to be printed
     * @return the tree as string at the given level
     */
    private String toString(int level) {

        StringBuilder text = new StringBuilder();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_Label)) {
                text.append(": null");
            } else {
                text.append(": ").append(m_ClassAttribute.value((int) m_Label));
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name()).append(" = ").append(m_Attribute.value(j));
                text.append(m_Children[j].toString(level + 1));
            }
        }
        return text.toString();
    }
    
    public double staticErrorEstimate(int N, int n, int k) {
        double E = (N - n + k - 1) / (double) (N + k);
        
        return E;
    }
    
    public double backUpError() {
        double E = 0;
        double totalInstances = DoubleStream.of(m_ClassDistribution).sum();
        for (MyJ48 child : m_Children) {
            double totalChildInstances = DoubleStream.of(child.m_ClassDistribution).sum();
            E += totalChildInstances/totalInstances
                * staticErrorEstimate((int) totalChildInstances, (int) child.m_ClassDistribution[(int) child.m_Label], child.m_ClassDistribution.length);
        }
        
        return E;
    }
}
