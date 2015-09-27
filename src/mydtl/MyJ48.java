package mydtl;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author susanti_2
 */
public class MyJ48 extends Classifier {
    
    private final double MISSING_VALUE = Double.NaN;

    /**
     * The node's children.
     */
    private MyID3[] m_Children;

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

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds Id3 tree classifier.
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

        makeTree(data);
    }

    /**
     * Creates an Id3 tree.
     *
     * @param data the training data
     * @exception Exception if tree failed to build
     */
    private void makeTree(Instances data) throws Exception {

        // Mengecek apakah tidak terdapat instance yang dalam node ini
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_Label = MISSING_VALUE;
            m_ClassDistribution = new double[data.numClasses()];
        } else {
            // Mencari IG maksimum
            double[] infoGains = new double[data.numAttributes()];
            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeGainRatio(data, att);
            }

            m_Attribute = data.attribute(Utils.maxIndex(infoGains));
            
            // Membuat daun jika IG-nya 0
            if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
                m_Attribute = null;

                m_ClassDistribution = new double[data.numClasses()];
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance inst = (Instance) data.instance(i);
                    m_ClassDistribution[(int) inst.classValue()]++;
                }

                Utils.normalize(m_ClassDistribution);
                m_Label = Utils.maxIndex(m_ClassDistribution);
                m_ClassAttribute = data.classAttribute();
            } else {
                // Membuat tree baru di bawah node ini
                Instances[] splitData = splitData(data, m_Attribute);
                m_Children = new MyID3[m_Attribute.numValues()];
                for (int j = 0; j < m_Attribute.numValues(); j++) {
                    m_Children[j] = new MyID3();
                    m_Children[j].makeTree(splitData[j]);
                }
            }
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

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, "
                + "please.");
        }
        if (m_Attribute == null) {
            return m_Label;
        } else {
            return m_Children[(int) instance.value(m_Attribute)].
                classifyInstance(instance);
        }
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
            throw new NoSupportForMissingValuesException("Id3: no missing values, "
                + "please.");
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
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }
    
    /**
     * Computes Gain Ratio for an attribute.
     *
     * @param data the data for which gain ratio is to be computed
     * @param att the attribute
     * @return the gain ratio for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeGainRatio(Instances data, Attribute att)
        throws Exception {

        return computeInfoGain(data, att) / computeSplitInformation(data, att);
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeInfoGain(Instances data, Attribute att)
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
    private double computeEntropy(Instances data) throws Exception {

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
    private double computeSplitInformation(Instances data,  Attribute att) throws Exception {
        
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
    private double log2(double num) {
        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }

    /**
     * split the dataset based on attribute
     *
     * @param data dataset used for splitting
     * @param att attribute used to split the dataset
     * @return 
     */
    private Instances[] splitData(Instances data, Attribute att) {

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

    /**
     * Adds this tree recursively to the buffer.
     *
     * @param id the unique id for the method
     * @param buffer the buffer to add the source code to
     * @return the last ID being used
     * @throws Exception if something goes wrong
     */
    protected int toSource(int id, StringBuffer buffer) throws Exception {
        int result;
        int i;
        int newID;
        StringBuffer[] subBuffers;

        buffer.append("\n");
        buffer.append("  protected static double node").append(id).append("(Object[] i) {\n");

        // leaf?
        if (m_Attribute == null) {
            result = id;
            if (Double.isNaN(m_Label)) {
                buffer.append("    return Double.NaN;");
            } else {
                buffer.append("    return ").append(m_Label).append(";");
            }
            if (m_ClassAttribute != null) {
                buffer.append(" // ").append(m_ClassAttribute.value((int) m_Label));
            }
            buffer.append("\n");
            buffer.append("  }\n");
        } else {
            buffer.append("    checkMissing(i, ").append(m_Attribute.index()).append(");\n\n");
            buffer.append("    // ").append(m_Attribute.name()).append("\n");

            // subtree calls
            subBuffers = new StringBuffer[m_Attribute.numValues()];
            newID = id;
            for (i = 0; i < m_Attribute.numValues(); i++) {
                newID++;

                buffer.append("    ");
                if (i > 0) {
                    buffer.append("else ");
                }
                buffer.append("if (((String) i[").append(m_Attribute.index()).append("]).equals(\"").append(m_Attribute.value(i)).append("\"))\n");
                buffer.append("      return node").append(newID).append("(i);\n");

                subBuffers[i] = new StringBuffer();
                newID = m_Children[i].toSource(newID, subBuffers[i]);
            }
            buffer.append("    else\n");
            buffer.append("      throw new IllegalArgumentException(\"Value '\" + i[").append(m_Attribute.index()).append("] + \"' is not allowed!\");\n");
            buffer.append("  }\n");

            // output subtree code
            for (i = 0; i < m_Attribute.numValues(); i++) {
                buffer.append(subBuffers[i].toString());
            }
            subBuffers = null;

            result = newID;
        }

        return result;
    }

    /**
     * Returns a string that describes the classifier as source. The classifier
     * will be contained in a class with the given name (there may be auxiliary
     * classes), and will contain a method with the signature:
     * <pre><code>
     * public static double classify(Object[] i);
     * </code></pre> where the array <code>i</code> contains elements that are
     * either Double, String, with missing values represented as null. The
     * generated code is public domain and comes with no warranty. <br/>
     * Note: works only if class attribute is the last attribute in the dataset.
     *
     * @param className the name that should be given to the source class.
     * @return the object source described by a string
     * @throws Exception if the source can't be computed
     */
    public String toSource(String className) throws Exception {
        StringBuffer result;
        int id;

        result = new StringBuffer();

        result.append("class ").append(className).append(" {\n");
        result.append("  private static void checkMissing(Object[] i, int index) {\n");
        result.append("    if (i[index] == null)\n");
        result.append("      throw new IllegalArgumentException(\"Null values "
            + "are not allowed!\");\n");
        result.append("  }\n\n");
        result.append("  public static double classify(Object[] i) {\n");
        id = 0;
        result.append("    return node").append(id).append("(i);\n");
        result.append("  }\n");
        toSource(id, result);
        result.append("}\n");

        return result.toString();
    }

    

  
}
