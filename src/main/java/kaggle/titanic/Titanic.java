package kaggle.titanic;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Rickard Edén (github.com/neph1)
 *
 * In words: https://www.kaggle.com/c/titanic/discussion/61841
 *
 */
public class Titanic {

    private static Logger log = LoggerFactory.getLogger(Titanic.class);

    public static void main(String[] args) throws IOException, InterruptedException {


        int numLinesToSkip = 1; // categories at the top
        char delimiter = ',';

        int labelIndex = 6;
        int numClasses = 2;     // survived or not
        int batchSize = 892;    //size of the dataset
        long seed = 12;
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("kaggle/titanic/train.csv").getFile()));

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle(25);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.9);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        final int numInputs = 6;
        int outputNum = 2;



        log.info("Build model....");
        MultiLayerNetwork model = getModel(seed, numInputs, outputNum);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        model.pretrain(iterator);
        for(int i=0; i< 10000 ; i++ ) {
            trainingData.shuffle(25);
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());


        recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("kaggle/titanic/test.csv").getFile()));
        iterator = new RecordReaderDataSetIterator(recordReader,418);
        DataSet verifyData = iterator.next();
        List<String> labelNames = new ArrayList<>();
        labelNames.add("Dead");
        labelNames.add("Alive");
        verifyData.setLabelNames(labelNames);
//        normalizer.transform(verifyData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
        output = model.output(verifyData.getFeatures());

        String outputPath = System.getProperty("user.dir")+"/src/main/resources/kaggle/titanic/output.csv";
        File outputFile = new File(outputPath);
        outputFile.createNewFile();
        FileWriter writer = null;
        try {
            writer = new FileWriter(outputFile);

        } catch (IOException e) {
            e.printStackTrace();
        }

        writer.write("PassengerId,Survived");
        writer.write("\n");

        for(int i = 0; i < output.rows(); i++){
            boolean alive = output.getRow(i).getFloat(1)  > output.getRow(i).getFloat(0);
            int id = verifyData.getFeatures().getRow(i).getInt(0);
            writer.write( id +","+ (alive ? 1 : 0) + "\n");
        }

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static MultiLayerNetwork getModel(long seed, int numInputs, int outputNum){
        int layerOne = 9;
        int layerTwo = 9;
        int layerThree = 4;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
//                .updater(new Sgd(0.18))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.17, 0.25))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(layerOne)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(layerOne).nOut(layerTwo)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(layerTwo).nOut(layerThree)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.SIGMOID)
                .nIn(layerThree).nOut(outputNum).build())
            //.backprop(true).pretrain(false)
            .build();

        return new MultiLayerNetwork(conf);
    }

}
