package com.example.alex.opencvdemo;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.ContentUris;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.PowerManager;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.view.View.OnClickListener;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.Rect;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.TimeUnit;

import io.reactivex.functions.Consumer;
import okhttp3.Call;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    static {
        if(!OpenCVLoader.initDebug())
        {
            Log.d("opencv","初始化失败");
        }
    }
    //首先还是先声明这个Spinner控件
    private Spinner apiSpinner;
    //定义一个String类型的List数组作为数据源
    private List<String> apiDataList;
    //定义一个ArrayAdapter适配器作为spinner的数据适配器
    private ArrayAdapter<String> apiAdapter;
    private  String selectedApi=null;

    private int detectModel=0;//0未知，1选择图片识别，2抓图识别，3实时识别
    private  boolean needChangeModel=false;
    private  Thread threadDetect=null;//定时执行的任务
    private boolean brunning=true;//标志是否在运行
    private Bitmap chooseImage=null;//选择图片
    private int detectIndex=0;//当前显示编号

    private  Bitmap cameraImage=null;//摄像头图片

    private CameraBridgeViewBase cameraView;
    private Classifier classifier=null;//识别类
    private static final String MODEL_FILE = "file:///android_asset/FacialExpressionReg.pb";
    private static final int IMAGE_SIZE = 48;

    private  MTCNN mtcnn;
    public  static  final  int CHOOSE_PHOTO=2;
    private CascadeClassifier cascadeClassifier = null; //级联分类器
    private int absoluteFaceSize = 0;
    private Handler detectHandleUpdate=null;
    private  int cameraIndex=CameraBridgeViewBase.CAMERA_ID_FRONT;
    private PowerManager.WakeLock mWakeLock;

    public static Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
         paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    //缩放图片,使用openCV，缩放方法采用area interpolation法
    private Bitmap scaleImage(Bitmap bitmap, int width, int height)
    {

        Mat src = new Mat();
        Mat dst = new Mat();
        Utils.bitmapToMat(bitmap, src);
        Imgproc.resize(src, dst, new Size(width,height),0,0,Imgproc.INTER_AREA);
        Bitmap bitmap1 = Bitmap.createBitmap(dst.cols(),dst.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dst, bitmap1);
        return bitmap1;
    }

    //Andorid不支持单通道图片，获取输入像素集
    public float[] getSingleChannelPixel(Bitmap bitmap) {
        float[] floatValues = new float[IMAGE_SIZE * IMAGE_SIZE * 1];

        if ((bitmap.getWidth() != IMAGE_SIZE) ||  (bitmap.getHeight() != IMAGE_SIZE)){
            Log.d("getSingleChannelPixel","获取像素时图片尺寸不对");
        }

        StringBuffer sBuffer = new StringBuffer("像素值：");
        for(int i = 0;i<bitmap.getWidth();i++)
        {
            for(int j =0;j<bitmap.getHeight();j++)
            {
                int col = bitmap.getPixel(i, j);
                int alpha = col&0xFF000000;
                int red = (col&0x00FF0000)>>16;
                int green = (col&0x0000FF00)>>8;
                int blue = (col&0x000000FF);
                int gray = (int)((float)red*0.3+(float)green*0.59+(float)blue*0.11);
                //int newColor = alpha|(gray<<16)|(gray<<8)|gray;
                floatValues[i + j* IMAGE_SIZE] = gray / 255.0f;
                sBuffer.append(gray) ;
                sBuffer.append(" ") ;
            }
        }
        //putStringToTxt(sBuffer.toString(), "pixel");
        return floatValues;
    }

    Bitmap adjustPhotoRotation(Bitmap bm, final int orientationDegree)
    {
        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);

        try {
            Bitmap bm1 = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), m, true);
            return bm1;
        } catch (OutOfMemoryError ex) {
        }
        return null;

    }
    //检测图片
    private String detectFaceEmotion(Bitmap bitmap,Rect[] facesArray)
    {
        if ( facesArray==null||facesArray.length==0)return  null;
        Bitmap destBitmap = Bitmap.createBitmap(bitmap, (int) (facesArray[0].tl().x), (int) (facesArray[0].tl().y), facesArray[0].width, facesArray[0].height);
        Bitmap scaleImage = scaleImage(destBitmap, 48, 48);
        Bitmap bitmap5 = toGrayscale(scaleImage);
        Bitmap bitmap6 = adjustPhotoRotation(bitmap5, 270);
        if (classifier==null){
            classifier = new Classifier(getAssets(),MODEL_FILE);
        }
        ArrayList<String> result = classifier.predict(getSingleChannelPixel(bitmap6));
        //0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
       String str = result.get(0);
        switch(str){
            case "0":
                str = "生气";break;
            case "1":
                str = "厌恶";break;
            case "2":
                str = "恐惧";break;
            case "3":
                str = "开心";break;
            case "4":
                str = "难过";break;
            case "5":
                str = "惊讶";break;
            case "6":
                str = "平静";break;
            default:
                Log.d("ccx","Tensorflow return is error.");;break;
        }
        return  str;
    }

    //检测人脸
    private  Rect[]  detectFaceRect(Bitmap bitmap){
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);

        Mat imgGray = new Mat();
        MatOfRect faces = new MatOfRect();

        if(img.empty())
        {
            Log.d("ccx","detectFace but img is empty");
            return null;
        }

        if(img.channels() ==3)
        {
            Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_RGB2GRAY);
        }
        else
        {
            imgGray = img;
        }

        cascadeClassifier.detectMultiScale(imgGray, faces, 1.1, 2, 2, new Size(absoluteFaceSize, absoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0){
            for (int i = 0; i < facesArray.length; i++) {
                facesArray[i].height+=facesArray[i].height*1/4;
                if (facesArray[i].height+facesArray[i].y>imgGray.height()){
                    facesArray[i].height=imgGray.height()-facesArray[i].y-1;
                }
                Imgproc.rectangle(imgGray, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
                Log.d("ccx","index:" + i + "topLeft:" + facesArray[i].tl() + "bottomRight:" + facesArray[i].br()+ "height:" + facesArray[i].height);
            }
        }else{
            return null;
        }
        Utils.matToBitmap(imgGray, bitmap);
        return  facesArray;
    }

    //检测人脸
    private  Rect[]  detectFaceRectByMTCNN(Bitmap bitmap){
        try {
            Vector<Box> boxes=mtcnn.detectFaces(bitmap,40);
            Mat img = new Mat();
            Utils.bitmapToMat(bitmap, img);
            ArrayList<Rect> rects=new ArrayList<Rect>();
            for (int i=0;i<boxes.size();i++) {
                android.graphics.Rect rect = boxes.get(i).transform2Rect();
                rect.top+=(rect.bottom-rect.top)/5;
                Rect cvRect = new Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
                if (cvRect.x<0){
                    cvRect.width+=cvRect.x;
                    cvRect.x=0;
                }
                if (cvRect.width+cvRect.x>bitmap.getWidth()){
                    cvRect.width=bitmap.getWidth()-cvRect.x;
                }

                if (cvRect.y<0){
                    cvRect.height+=cvRect.y;
                    cvRect.y=0;
                }
                if (cvRect.height+cvRect.y>bitmap.getHeight()){
                    cvRect.height=bitmap.getHeight()-cvRect.y;
                }
                rects.add(cvRect);
                Imgproc.rectangle(img, cvRect.tl(), cvRect.br(), new Scalar(0, 255, 0, 255), 3);
            }
            Utils.matToBitmap(img, bitmap);
            Rect[] rectarr=new Rect[rects.size()];
            return  rects.toArray(rectarr);
        }catch (Exception e){
            Log.e("facerect","detectFaceRectByMTCNN:",e);
        }
        return null;
    }


    private  String detectFaceEmotion(Bitmap bitmap){
        try{
            if(selectedApi.equals("Face++")){
                return  detectFaceEmotionByFaceCPP(bitmap);
            }else  if (selectedApi.equals("微软Azure")){
                return  detectFaceEmotionByMicroAzure(bitmap);
            }else{
                Rect[] facesArray=detectFaceRectByMTCNN(bitmap);
                String ret =  detectFaceEmotion(bitmap,facesArray);
                Mat mat=new Mat();
                Utils.bitmapToMat(bitmap,mat);
                Imgproc.putText(mat, "Inner", new Point(5, 40), 3, 1, new Scalar(0, 255, 0, 255), 2);
                Utils.matToBitmap(mat,bitmap);
                return  ret;
            }
        }catch (Exception ex){
            Log.e("facedetect","detectFaceEmotion:",ex);
        }
        return null;
    }
    private  String detectResult=null;
    private String detectFaceEmotionByFaceCPP(Bitmap bitmap){
        Mat mat=new Mat();
        Utils.bitmapToMat(bitmap,mat);
        try{
            detectResult=null;
            OkHttpClient client = new OkHttpClient();
            // form 表单形式上传
            MultipartBody.Builder requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM);
            requestBody.addFormDataPart("api_key","six6aKZ9f6pNupm9XZ4HqOsoeSDCRngu");
            requestBody.addFormDataPart("api_secret","6MrBjWsqZXcwT04L-Z99Iah1zF3qUpLi");
            requestBody.addFormDataPart("return_attributes","emotion");
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
            // MediaType.parse() 里面是上传的文件类型。
            RequestBody body = RequestBody.create(MediaType.parse("image/*"), baos.toByteArray());
            // 参数分别为， 请求key ，文件名称 ， RequestBody
            requestBody.addFormDataPart("image_file", "imagefile", body);
            Request request = new Request.Builder().url("https://api-cn.faceplusplus.com/facepp/v3/detect").post(requestBody.build()).tag(null).build();
            // readTimeout("请求超时时间" , 时间单位);
            Call call = client.newBuilder().readTimeout(8000, TimeUnit.MILLISECONDS).build().newCall(request);
            try {
                Response response=call.execute();
                if (response.isSuccessful()){
                    String str = response.body().string();
                    Log.i("FACE++", response.message() + " , body " + str);
                    /*
                    {
                      "image_id": "B7B3u1yn5FThJtP2uDnJtQ==",
                      "request_id": "1542190044,47db8847-263b-42fe-8e26-0c4f646b2446",
                      "time_used": 228,
                      "faces": [
                        {
                          "attributes": {
                            "emotion": {
                              "sadness": 0.07,
                              "neutral": 6.487,
                              "disgust": 0.452,
                              "anger": 0.034,
                              "surprise": 0.059,
                              "fear": 15.858,
                              "happiness": 77.04
                            }
                          },
                          "face_rectangle": {
                            "width": 58,
                            "top": 49,
                            "left": 54,
                            "height": 58
                          },
                          "face_token": "4733a62474eeffecc9e254edbca97661"
                        }
                      ]
                    }
                     */
                    try {
                        JSONObject jsonObject=new JSONObject(str);
                        JSONObject resObject = jsonObject.getJSONArray("faces").getJSONObject(0);
                        JSONObject emotion = resObject.getJSONObject("attributes").getJSONObject("emotion");
                        JSONObject face_rectangle = resObject.getJSONObject("face_rectangle");
                        double sadness=emotion.getDouble("sadness");
                        double neutral=emotion.getDouble("neutral");
                        double disgust=emotion.getDouble("disgust");
                        double anger=emotion.getDouble("anger");
                        double surprise=emotion.getDouble("surprise");
                        double fear=emotion.getDouble("fear");
                        double happiness=emotion.getDouble("happiness");
                        double ret = Math.max(Math.max(Math.max(Math.max(Math.max(Math.max(sadness,neutral),disgust),anger),surprise),fear),happiness);
                        if ( ret == sadness){
                            detectResult="难过";
                        }else if (ret==neutral){
                            detectResult="平静";
                        }
                        else if (ret==disgust){
                            detectResult="厌恶";
                        }
                        else if (ret==anger){
                            detectResult="生气";
                        }
                        else if (ret==surprise){
                            detectResult="惊讶";
                        }
                        else if (ret==surprise){
                            detectResult="恐惧";
                        }
                        else if (ret==happiness){
                            detectResult="高兴";
                        }
                        double left=face_rectangle.getDouble("left");
                        double top=face_rectangle.getDouble("top");
                        double width=face_rectangle.getDouble("width");
                        double height=face_rectangle.getDouble("height");
                        Imgproc.rectangle(mat,new Point(left,top),new Point(left+width,top+height),new Scalar(0, 255, 0, 255), 3);
                    }catch (Exception ex){
                        Log.e("FACE++", "onResponse: ",ex );
                    }
                }

            } catch (IOException e) {
                Log.e("Face++", "detectFaceEmotionByFaceCPP json: ", e);
            }
        }catch (Exception ex){
            Log.e("Face++", "detectFaceEmotionByFaceCPP: ", ex);
        }
        Imgproc.putText(mat, "Face++", new Point(5, 40), 3, 1, new Scalar(0, 255, 0, 255), 2);
        Utils.matToBitmap(mat,bitmap);
        return detectResult;
    }
    private String detectFaceEmotionByMicroAzure(Bitmap bitmap){
        Mat mat=new Mat();
        Utils.bitmapToMat(bitmap,mat);
        try{
            detectResult=null;
            OkHttpClient client = new OkHttpClient();

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);

            RequestBody body = RequestBody.create(MediaType.parse("application/octet-stream"), baos.toByteArray());
            Request.Builder builder = new Request.Builder().url("https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceAttributes=emotion");
            builder.addHeader("Ocp-Apim-Subscription-Key","c8b7c760b42b49adbb1444e42063ff49");
            builder.addHeader("Content-Type","application/octet-stream");

            Request request = builder.post(body).tag(null).build();

            // readTimeout("请求超时时间" , 时间单位);
            Call call = client.newBuilder().readTimeout(8000, TimeUnit.MILLISECONDS).build().newCall(request);
            try {
                Response response=call.execute();
                if (response.isSuccessful()){
                    String str = response.body().string();
                    Log.i("Azure", response.message() + " , body " + str);
                    /*
                    [
                          {
                            "faceId": "c1e7830b-d2da-4223-8e43-5b62f62ed20f",
                            "faceRectangle": {
                              "top": 47,
                              "left": 41,
                              "width": 63,
                              "height": 63
                            },
                            "faceAttributes": {
                              "emotion": {
                                "anger": 0.001,
                                "contempt": 0.001,
                                "disgust": 0,
                                "fear": 0,
                                "happiness": 0.002,
                                "neutral": 0.914,
                                "sadness": 0.082,
                                "surprise": 0.001
                              }
                            }
                          }
                        ]
                     */
                    try {
                        JSONArray jsonArray=new JSONArray(str);
                        if (jsonArray.length()==0)return detectResult;
                        JSONObject jsonObject=jsonArray.getJSONObject(0);
                        JSONObject emotion = jsonObject.getJSONObject("faceAttributes").getJSONObject("emotion");
                        JSONObject face_rectangle = jsonObject.getJSONObject("faceRectangle");
                        double contempt=emotion.getDouble("contempt");
                        double sadness=emotion.getDouble("sadness");
                        double neutral=emotion.getDouble("neutral");
                        double disgust=emotion.getDouble("disgust");
                        double anger=emotion.getDouble("anger");
                        double surprise=emotion.getDouble("surprise");
                        double fear=emotion.getDouble("fear");
                        double happiness=emotion.getDouble("happiness");
                        double ret =Math.max(Math.max(Math.max(Math.max(Math.max(Math.max(Math.max(sadness,neutral),disgust),anger),surprise),fear),happiness),contempt);
                        if (ret==contempt){
                            detectResult="藐视";
                        }
                        else if ( ret == sadness){
                            detectResult="难过";
                        }else if (ret==neutral){
                            detectResult="平静";
                        }
                        else if (ret==disgust){
                            detectResult="厌恶";
                        }
                        else if (ret==anger){
                            detectResult="生气";
                        }
                        else if (ret==surprise){
                            detectResult="惊讶";
                        }
                        else if (ret==surprise){
                            detectResult="恐惧";
                        }
                        else if (ret==happiness){
                            detectResult="高兴";
                        }
                        double left=face_rectangle.getDouble("left");
                        double top=face_rectangle.getDouble("top");
                        double width=face_rectangle.getDouble("width");
                        double height=face_rectangle.getDouble("height");

                        Imgproc.rectangle(mat,new Point(left,top),new Point(left+width,top+height),new Scalar(0, 255, 0, 255), 3);

                    }catch (Exception ex){
                        Log.e("Azure", "onResponse: ",ex );
                    }
                }else{
                    Log.e("Azure", "onResponse: "+response.body().string());
                }

            } catch (IOException e) {
                Log.e("Azure", "detectFaceEmotionByMicroAzure json: ", e);
            }
        }catch (Exception ex){
            Log.e("Azure", "detectFaceEmotionByMicroAzure: ", ex);
        }
        Imgproc.putText(mat, "Micro Azure", new Point(5, 40), 3, 1, new Scalar(0, 255, 0, 255), 2);
        Utils.matToBitmap(mat,bitmap);
        return detectResult;
    }

    private void initializeOpenCVDependencies() {
        try {
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface_improved);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface_improved.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // 加载cascadeClassifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("opencv","Error loading cascade");
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }
    @Override
    protected void onResume() {
        super.onResume();
        if (mWakeLock != null) {
            mWakeLock.acquire();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mWakeLock != null) {
            mWakeLock.release();
        }
    }
    private  Mat matLin=new Mat();//临时图像对象
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat temp = inputFrame.rgba();
        int w = temp.width();
        int h = temp.height();
        try {
            if (cameraIndex==CameraBridgeViewBase.CAMERA_ID_FRONT){
                Core.transpose(temp, matLin);
                Core.flip(matLin, temp, 1);
                //转置函数,将图像顺时针顺转（对换）
                Core.flip(temp, matLin, 0);
            }else{
                Core.transpose(temp, matLin);
                Core.flip(matLin, temp, 1);
                matLin=temp;
            }

        } catch (Exception ex) {
            Log.e("frame", ex.getMessage(), ex);
        }
        if (!needChangeModel) {
            switch (detectModel) {
                case 2:
                case 3:
                    try {
                        Bitmap bitmap = Bitmap.createBitmap(matLin.width(), matLin.height(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(matLin, bitmap);
                        cameraImage = bitmap;
                        if (detectModel == 2) {
                            detectModel = 98;
                        }
                        if (detectModel == 3) {
                            detectModel = 99;
                        }
                    } catch (Exception ex) {
                        Log.e("frame", ex.getMessage(), ex);
                    }
                    break;
                default:
                    break;
            }
        }
        if (matLin.width() != w || matLin.height() != h) {
            int new_w,new_h;
            if ((matLin.width()/(double)matLin.height())>(w/(double)h)) {
                new_h = w * matLin.height() / matLin.width();
                new_w = w;
            }else {
                new_h = h;
                new_w = h * matLin.width() / matLin.height();
            }
            Imgproc.resize(matLin,temp,new Size(new_w,new_h),0,0,Imgproc.INTER_AREA);
        }else{
            temp=matLin;
        }
        return temp;
    }

    private void doExcuteDetect(){
        try{
            switch (detectModel){
                case 1:
                {
                    if ( null == chooseImage )return;
                    String ret =  detectFaceEmotion(chooseImage);
                    displayResult(chooseImage,ret);
                    chooseImage=null;
                }
                break;
                case 98:
                case 99:
                {
                    if (null==cameraImage)return;
                    String ret =  detectFaceEmotion(cameraImage);
                    displayResult(cameraImage,ret);
                    cameraImage=null;
                     if (detectModel==99){
                        detectModel=3;
                    }
                }
                break;
            }
        }catch (Exception ex){
            Log.e("Excute", "doExcuteDetect: ", ex);
        }
        finally {
            cameraImage=null;
            chooseImage=null;
        }
    }
    @Override
    public void onDestroy() {
        super.onDestroy();
        try{
            brunning=false;
            if (cameraView != null) {
                cameraView.disableView();
            }

            if (threadDetect!=null&&threadDetect.isAlive()){
                Thread.sleep(100);
                threadDetect.interrupt();
            }
        }catch (Exception ex){
            Log.e("Exit", "onDestroy: ",ex );
        }
    }

    class MyClickListener implements OnClickListener{
        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub
            try{
                switch (v.getId()) {
                    case  R.id.choose_image:
                        if (cameraImage!=null||chooseImage!=null){
                            needChangeModel=true;
                            Thread.sleep(200);
                            if (cameraImage!=null||chooseImage!=null) {
                                return;
                            }
                        }
                        detectModel=1;
                        needChangeModel=false;
                        findViewById(R.id.choose_image).setSelected(true);
                        findViewById(R.id.capture_image).setSelected(false);
                        findViewById(R.id.real_image).setSelected(false);
                        openAlbum();
                        break;
                    case R.id.capture_image:
                        if (cameraImage!=null||chooseImage!=null){
                            needChangeModel=true;
                            Thread.sleep(200);
                            if (cameraImage!=null||chooseImage!=null) {
                                return;
                            }
                        }
                        detectModel=2;
                        needChangeModel=false;
                        findViewById(R.id.choose_image).setSelected(false);
                        findViewById(R.id.capture_image).setSelected(true);
                        findViewById(R.id.real_image).setSelected(false);
                        break;
                    case  R.id.real_image:
                        if (cameraImage!=null||chooseImage!=null){
                            needChangeModel=true;
                            Thread.sleep(200);
                            if (cameraImage!=null||chooseImage!=null) {
                                return;
                            }
                        }
                        detectModel=3;
                        needChangeModel=false;
                        findViewById(R.id.choose_image).setSelected(false);
                        findViewById(R.id.capture_image).setSelected(false);
                        findViewById(R.id.real_image).setSelected(true);
                        break;
                    case  R.id.camera_select:
                        {
                            if (cameraIndex==CameraBridgeViewBase.CAMERA_ID_FRONT){
                                cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                                cameraIndex=CameraBridgeViewBase.CAMERA_ID_BACK;
                                ((Button)findViewById(R.id.camera_select)).setText("前置相机");
                            }else{
                                cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                                cameraIndex=CameraBridgeViewBase.CAMERA_ID_FRONT;
                                ((Button)findViewById(R.id.camera_select)).setText("后置相机");
                            }
                            cameraView.disableView();
                            cameraView.enableView();
                        }
                        break;
                    default:
                        break;
                }
            }catch (Exception ex){

            }
        }
    }
    private void openAlbum(){
        Intent intent = new Intent("android.intent.action.GET_CONTENT");
        intent.setType("image/*");
        startActivityForResult(intent,CHOOSE_PHOTO);//打开相册
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode){
            case CHOOSE_PHOTO:
                if (resultCode == RESULT_OK){
                    //判断手机系统版本号
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT){
                        //4.4及以上系统使用这个方法处理图片
                        handleImageOnKitKat(data);
                    }else {
                        //4.4以下系统使用这个放出处理图片
                        handleImageBeforeKitKat(data);
                    }
                }
                break;
            default:
                break;
        }
    }

    @TargetApi(Build.VERSION_CODES.KITKAT)
    private void handleImageOnKitKat(Intent data){
        String imagePath = null;
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(this,uri)){
            //如果是document类型的Uri,则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())){
                String id = docId.split(":")[1];//解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,selection);
            }else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())){
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"),Long.valueOf(docId));
                imagePath = getImagePath(contentUri,null);
            }
        }else if ("content".equalsIgnoreCase(uri.getScheme())){
            //如果是content类型的Uri，则使用普通方式处理
            imagePath = getImagePath(uri,null);
        }else if ("file".equalsIgnoreCase(uri.getScheme())){
            //如果是file类型的Uri，直接获取图片路径即可
            imagePath = uri.getPath();
        }
        chooseImage =  createImage(imagePath);//根据图片路径显示图片
    }

    private void handleImageBeforeKitKat(Intent data){
        Uri uri = data.getData();
        String imagePath = getImagePath(uri,null);
        chooseImage =  createImage(imagePath);
    }

    private String getImagePath(Uri uri,String selection){
        String path = null;
        //通过Uri和selection来获取真实的图片路径
        Cursor cursor = getContentResolver().query(uri,null,selection,null,null);
        if (cursor != null){
            if (cursor.moveToFirst()){
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }

    private  void  displayResult(final Bitmap bitmap,final String result){
        detectHandleUpdate.post(new Runnable() {
            @Override
            public void run() {
                GridLayout layout = findViewById(R.id.detect_result);
                LinearLayout child =  (LinearLayout)layout.getChildAt(detectIndex);
                detectIndex = (detectIndex+1)%layout.getChildCount();
                ImageView iv =(ImageView)child.getChildAt(0);
                iv.setImageBitmap(bitmap);
                TextView tv = (TextView)child.getChildAt(1);
                if (result==null)
                    tv.setText("未知");
                else tv.setText(result);
            }
        });
    }

    private Bitmap createImage(String imagePath){
        if (imagePath != null){
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap.getWidth()>350||bitmap.getHeight()>350){
                double ratio=bitmap.getWidth()/(double)bitmap.getHeight();
                int w=350;
                int h=350;
                if (ratio>1){
                    h=w*bitmap.getHeight()/bitmap.getWidth();
                }else{
                    w=h*bitmap.getWidth()/bitmap.getHeight();
                }
                Mat mat=new Mat();
                Mat dst=new Mat();
                Utils.bitmapToMat(bitmap,mat);
                Imgproc.resize(mat,dst,new Size(w,h));
                Utils.matToBitmap(mat,bitmap);
            }
            return bitmap;
        }else {
            Toast.makeText(this,"failed to get iamge",Toast.LENGTH_SHORT).show();
        }
        return  null;
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);

        mtcnn=new MTCNN(getAssets());

        Button chooseBtn = findViewById(R.id.choose_image);
        Button captureBtn = findViewById(R.id.capture_image);
        Button realBtn = findViewById(R.id.real_image);
        Button cameraSelect=findViewById(R.id.camera_select);
        chooseBtn.setOnClickListener(new MyClickListener());
        captureBtn.setOnClickListener(new MyClickListener());
        realBtn.setOnClickListener(new MyClickListener());
        cameraSelect.setOnClickListener(new MyClickListener());

        cameraView =findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        initializeOpenCVDependencies();
        detectHandleUpdate=new Handler();

        apiSpinner=findViewById(R.id.api_spin);
        //为dataList赋值，将下面这些数据添加到数据源中
        apiDataList = new ArrayList<String>();
        apiDataList.add("内置算法");
        apiDataList.add("Face++");
        apiDataList.add("微软Azure");

        apiAdapter = new ArrayAdapter<String>(this,android.R.layout.simple_spinner_item,apiDataList);

        //为适配器设置下拉列表下拉时的菜单样式。
        apiAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        //为spinner绑定我们定义好的数据适配器
        apiSpinner.setAdapter(apiAdapter);
        apiSpinner.setSelection(1);
//为spinner绑定监听器，这里我们使用匿名内部类的方式实现监听器
        apiSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedApi=apiDataList.get(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                selectedApi=null;
            }
        });

        //初始化检测线程
        threadDetect=new Thread(new Runnable() {
            @Override
            public void run() {
                while (brunning){
                    try{
                        Thread.sleep(100);
                        doExcuteDetect();
                    }catch (Exception ex){
                        Log.e("Thread", "detect run: ", ex);
                    }
                }
            }
        });
        threadDetect.start();

        PowerManager powerManager = (PowerManager)getSystemService(POWER_SERVICE);
        if (powerManager != null) {
            mWakeLock = powerManager.newWakeLock(PowerManager.FULL_WAKE_LOCK,getClass().getName());
        }

        //权限请求
        RxPermissions rxPermission = new RxPermissions(this);
        rxPermission.requestEach(Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.INTERNET)
                .subscribe(new Consumer<Permission>() {
                    @Override
                    public void accept(Permission permission) throws Exception {
                        if (permission.granted) {
                            // 用户已经同意该权限
                            Log.d("permission", permission.name + " is granted.");
                            if(permission.name.equals(Manifest.permission.CAMERA)){
                                cameraView.enableView();
                            }
                        } else if (permission.shouldShowRequestPermissionRationale) {
                            // 用户拒绝了该权限，没有选中『不再询问』（Never ask again）,那么下次再次启动时，还会提示请求权限的对话框
                            Log.d("permission", permission.name + " is denied. More info should be provided.");
                        } else {
                            // 用户拒绝了该权限，并且选中『不再询问』
                            Log.d("permission", permission.name + " is denied.");
                        }
                    }
                });

    }
}