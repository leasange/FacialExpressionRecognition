package com.example.alex.opencvdemo;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.lang.Math;

public class Classifier {

    //模型中输入变量的名称
    private static final String inputName = "input_x";
    //模型中输出变量的名称
    private static final String outputName = "predict";
    //概率变量的名称
    private static final String probabilityName = "probability";
    //cnn输出层的数据
    private static final String outlayerName = "outlayer";
    //图片维度
    private static final int IMAGE_SIZE = 48;

    TensorFlowInferenceInterface inferenceInterface;


    static {
        //加载libtensorflow_inference.so库文件
        System.loadLibrary("tensorflow_inference");
        Log.e("tensorflow","libtensorflow_inference.so库加载成功");
    }
    Classifier(AssetManager assetManager, String modePath) {
        //初始化TensorFlowInferenceInterface对象
        inferenceInterface = new TensorFlowInferenceInterface(assetManager,modePath);
        Log.e("tf","TensoFlow模型文件加载成功");
    }

    public ArrayList predict(float[] inputdata)
    {
        ArrayList<String> list = new ArrayList<>();

        inferenceInterface.feed(inputName, inputdata, new long[]{IMAGE_SIZE * IMAGE_SIZE});
        String[] outputNames = new String[]{outputName, probabilityName, outlayerName};
        inferenceInterface.run(outputNames);

        float[] outlayer = new float[7];
        inferenceInterface.fetch(outlayerName, outlayer);

        int[] labels = new int[1];
        inferenceInterface.fetch(outputName,labels);
        int label = labels[0];
        float[] prob = new float[7];
        inferenceInterface.fetch(probabilityName, prob);

        DecimalFormat df = new DecimalFormat("0.000000");
        float label_prob = prob[label];
        list.add(Integer.toString(label));
        list.add(df.format(label_prob));

        return list;
    }

}
