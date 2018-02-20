/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.citec.sc.embeddingservice;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import spark.Spark;

/**
 *
 * @author sherzod
 */
public class StartService {

    private static final String filePathEN = "embeddingEN.txt";
    private static final String filePathES = "embeddingES.txt";
    private static final String filePathDE = "embeddingDE.txt";
    private static final String filePathGloveEN = "glove.6B.300d.txt";
    
    private static Word2Vec vecEN = null;
    private static Word2Vec vecDE = null;
    private static Word2Vec vecES = null;

    public static void main(String[] args) throws Exception {
//
//        args = new String[2];
//        args[0] = "-s";
//        args[1] = "api";

        if (args != null) {
            if (args[1].equals("api")) {
                vecEN = loadModel(filePathEN);
                vecDE = loadModel(filePathDE);
                vecES = loadModel(filePathES);

                int port = 8081;
                Spark.port(port);

                System.out.println("Starting the service with port :" + port);

                Spark.get("/similarity", "application/json", (request, response) -> {
                    //get input from client
                    String word1 = request.queryParams("word1");
                    String word2 = request.queryParams("word2");
                    String lang = request.queryParams("lang");

                    try {
                        word1 = URLDecoder.decode(word1, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }
                    try {
                        word2 = URLDecoder.decode(word2, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }

                    double similarity = 0;

                    switch (lang) {
                        case "EN":
                            similarity = computeSimilarity(word1, word2, vecEN);
                            break;
                        case "DE":
                            similarity = computeSimilarity(word1, word2, vecDE);
                            break;
                        case "ES":
                            similarity = computeSimilarity(word1, word2, vecES);
                            break;
                    }

                    return similarity;

                }, new JsonTransformer());
                
                Spark.get("/word2vec", "application/json", (request, response) -> {
                    //get input from client
                    String word = request.queryParams("word");
                    String lang = request.queryParams("lang");

                    try {
                        word = URLDecoder.decode(word, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }
                    

                    double[] vec =  new double[0];

                    switch (lang) {
                        case "EN":
                            vec = word2vec(word, vecEN);
                            break;
                        case "DE":
                            vec = word2vec(word, vecDE);
                            break;
                        case "ES":
                            vec = word2vec(word, vecES);
                            break;
                    }

                    return vec;

                }, new JsonTransformer());
            } 
            else if (args[1].equals("gloveApi")) {
                vecEN = loadModel(filePathGloveEN);

                int port = 8081;
                Spark.port(port);

                System.out.println("Starting the service with port :" + port);

                Spark.get("/similarity", "application/json", (request, response) -> {
                    //get input from client
                    String word1 = request.queryParams("word1");
                    String word2 = request.queryParams("word2");
                    String lang = request.queryParams("lang");

                    try {
                        word1 = URLDecoder.decode(word1, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }
                    try {
                        word2 = URLDecoder.decode(word2, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }

                    double similarity = computeSimilarity(word1, word2, vecEN);

                    return similarity;

                }, new JsonTransformer());
                
                Spark.get("/word2vec", "application/json", (request, response) -> {
                    //get input from client
                    String word = request.queryParams("word");
                    String lang = request.queryParams("lang");

                    try {
                        word = URLDecoder.decode(word, "UTF-8");
                    } catch (UnsupportedEncodingException ex) {
                    }
                    

                    double[] vec = word2vec(word, vecEN);
                    
                    return vec;

                }, new JsonTransformer());
            }
            
            else if (args[1].equals("train")) {
                System.out.println("Training word2vec model");

                List<String> languages = new ArrayList<>();
                languages.add("EN");
                languages.add("DE");
                languages.add("ES");

                for (String lang : languages) {

                    String sentenceDir = "sentenceCorpus" + lang;

                    File dir = new File(sentenceDir);

                    if (dir.exists()) {

                        System.out.println("Training the model for " + lang);
                        Word2Vec vec = createModel(sentenceDir);

                        System.out.println("Vocab size: " + vec.getVocab().numWords());

                        String modelFilePath = "embedding" + lang + ".txt";
                        saveModel(vec, modelFilePath);
                    }
                }
            }
        }
    }

    private static Word2Vec loadModel(String filePath) throws FileNotFoundException {
        Word2Vec word2Vec = null;

        System.out.println("Loading model from " + filePath);
        File file = new File(filePath);
        if (file.exists()) {
            word2Vec = WordVectorSerializer.readWord2VecModel(filePath);
        }
        return word2Vec;
    }
    
    private static double computeSimilarity(String word1, String word2, Word2Vec vec) {
        double sim = 0;

        if (vec == null) {
            return sim;
        }

        word1 = word1.toLowerCase();
        word2 = word2.toLowerCase();
        
        Collection<String> words = vec.getVocab().words();

        if (word1.contains(" ") || word2.contains(" ")) {

            double[] vec1 = summedVector(word1, vec);
            double[] vec2 = summedVector(word2, vec);

            sim = cosineSimilarity(vec1, vec2);

        } else {
            try {

                sim = vec.similarity(word1, word2);

            } catch (Exception e) {

            }
        }

        return sim;
    }

    private static double[] summedVector(String w, Word2Vec vec) {

        double[] summedVec = new double[vec.getLayerSize()];

        //initialize with 0s
        for (int i = 0; i < summedVec.length; i++) {
            summedVec[i] = 0;
        }

        String[] splitWords = w.split(" ");

        for (String s : splitWords) {
            double[] wordVec = vec.getWordVector(s);

            if (wordVec == null) {
                continue;
            }

            for (int i = 0; i < summedVec.length; i++) {
                summedVec[i] += wordVec[i];
            }
        }

        return summedVec;
    }
    
    private static double[] word2vec(String w, Word2Vec vec) {
        
        double[] summedVec = null;
        
        if(w.contains(" ")){
            summedVec = summedVector(w, vec);
        }
        else{
            summedVec = vec.getWordVector(w);
        }

        return summedVec;
    }

    private static double cosineSimilarity(double[] vec1, double[] vec2) {
        double sim = 0;

        //dot product
        for (int i = 0; i < vec1.length; i++) {
            sim += vec1[i] * vec2[i];
        }

        double norm1 = 0;
        double norm2 = 0;

        //calculate the detominator of the equation
        for (int i = 0; i < vec1.length; i++) {
            norm1 += Math.pow(vec1[i], 2);
        }
        norm1 = Math.sqrt(norm1);

        for (int i = 0; i < vec2.length; i++) {
            norm2 += Math.pow(vec2[i], 2);
        }
        norm2 = Math.sqrt(norm2);

        sim = sim / (norm1 * norm2);

        if (Double.isInfinite(sim) || Double.isNaN(sim)) {
            sim = 0;
        }

        return sim;
    }

    private static Word2Vec createModel(String dir) {

        SentenceIterator iter = new FileSentenceIterator(new File(dir));
        TokenizerFactory t = new DefaultTokenizerFactory(); // new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 3);
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec word2Vec = new Word2Vec.Builder()
                .negativeSample(0.1)
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(300)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        System.out.println("Fitting Word2Vec model....");
        word2Vec.fit();

        return word2Vec;
    }

    private static void visualize(Word2Vec word2Vec) {
        System.out.println("Visualising the embeddings ... ");
        System.out.println("Vocabulary size: " + word2Vec.getVocab().numWords());

        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .build();
        word2Vec.lookupTable().plotVocab(tsne, 100, new File("plot.txt"));
    }

    private static void saveModel(Word2Vec vec, String filePath) {

        System.out.println("Saving model to " + filePath);
        WordVectorSerializer.writeWord2VecModel(vec, filePath);
    }

}
