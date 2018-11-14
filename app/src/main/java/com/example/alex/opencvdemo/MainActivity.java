package com.example.alex.opencvdemo;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.widget.Button;
import android.view.View.OnClickListener;
import android.widget.TextView;

import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.Rect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.*;

import io.reactivex.functions.Consumer;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    static {
        if(!OpenCVLoader.initDebug())
        {
            Log.d("opencv","初始化失败");
        }
    }

    private int detectModel=0;//0未知，1选择图片识别，2抓图识别，3实时识别
    private  Thread threadDetect;//定时执行的任务
    private boolean brunning=true;//标志是否在运行
    private Uri chooseImageUrl;//选择图片地址

    private CameraBridgeViewBase cameraView;
    private TextView reslutTextView;
    private Classifier classifier;//识别类
    private static final String MODEL_FILE = "file:///android_asset/FacialExpressionReg.pb";
    private static final int IMAGE_SIZE = 48;

    public static final int TAKE_PHOTO = 1;
    private CascadeClassifier cascadeClassifier = null; //级联分类器
    private int absoluteFaceSize = 0;
    Mat  mRgba;
    Mat  mGray;

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

    public static void putStringToTxt(String s, String name)
    {
        try
        {
            FileOutputStream outStream = new FileOutputStream("/sdcard/"+name+"cc.txt",true);
            OutputStreamWriter writer = new OutputStreamWriter(outStream,"gb2312");
            writer.write(s);
            writer.write("/n");
            writer.flush();
            writer.close();
            outStream.close();
        }
        catch (Exception e)
        {
            Log.e("m", "file write error");
        }
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
    private  String faceResult="";
    private void detectFace(Bitmap bitmap)
    {
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);

        Mat imgGray = new Mat();;
        MatOfRect faces = new MatOfRect();

        if(img.empty())
        {
            Log.d("ccx","detectFace but img is empty");
            return;
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
                Imgproc.rectangle(imgGray, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
                Log.d("ccx","index:" + i + "topLeft:" + facesArray[i].tl() + "bottomRight:" + facesArray[i].br()+ "height:" + facesArray[i].height);
            }
        }else{
            return;
        }

        Utils.matToBitmap(imgGray, bitmap);
        //imageView.setImageBitmap(bitmap);
        Bitmap destBitmap = Bitmap.createBitmap(bitmap, (int) (facesArray[0].tl().x), (int) (facesArray[0].tl().y), facesArray[0].width, facesArray[0].height);
        Bitmap scaleImage = scaleImage(destBitmap, 48, 48);
        Bitmap bitmap5 = toGrayscale(scaleImage);
        Bitmap bitmap6 = adjustPhotoRotation(bitmap5, 270);

        classifier = new Classifier(getAssets(),MODEL_FILE);
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
        faceResult=str;
        reslutTextView.setText("识别结果: " + faceResult);
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

    private  Mat matLin=new Mat();//临时图像对象
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
       // mGray = inputFrame.gray();
        /*
        try{
            Core.transpose(mRgba, matLin);
         //  int h = cameraView.getHeight()/cameraView.getWidth()*mRgba.height();
         //  int w =mRgba.width();
            Core.flip(matLin, mRgba, 1);
            //转置函数,将图像顺时针顺转（对换）
            Core.flip(mRgba, matLin, 0);
            mRgba = matLin;

           // Core.flip(mRgba, mRgba, 1);//flip aroud Y-axis
            //  Core.flip(mRgba, mRgba, 1);//flip aroud Y-axis
            //  Core.flip(mGray, mGray, 1);
            Bitmap bitmap=Bitmap.createBitmap(mRgba.width(), mRgba.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mRgba,bitmap);
            detectFace(bitmap);
            Utils.bitmapToMat(bitmap,mRgba);
          //  Imgproc.resize(mRgba,mRgba, new Size(w,h), 0.0D, 0.0D, 0); //将转置后的图像缩放为mRgbaF的大小
        }catch (Exception ex){
            Log.e("frame",ex.getMessage(),ex);
        }*/
        return mRgba;
    }

    private void doExcuteDetect(){
        switch (detectModel){
            case 1:
            {
                if (chooseImageUrl==null)return;

            }
                break;
            case 2:
                break;
            case 3:
                break;
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
    };

    class MyClickListener implements OnClickListener{
        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub
            switch (v.getId()) {
                case  R.id.choose_image:
                    detectModel=1;
                    break;
                case R.id.capture_image:
                    detectModel=2;
                    break;
                case  R.id.real_image:
                    detectModel=3;
                    break;
                default:
                    break;
            }
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);

        Button chooseBtn = (Button)findViewById(R.id.choose_image);
        Button captureBtn = (Button)findViewById(R.id.capture_image);
        Button realBtn = (Button)findViewById(R.id.real_image);
        chooseBtn.setOnClickListener(new MyClickListener());
        captureBtn.setOnClickListener(new MyClickListener());
        realBtn.setOnClickListener(new MyClickListener());

        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        initializeOpenCVDependencies();

        //初始化检测线程
        threadDetect=new Thread(new Runnable() {
            @Override
            public void run() {
                while (brunning){
                    try{
                        Thread.sleep(100);
                        doExcuteDetect();
                    }catch (Exception ex){

                    }
                }
            }
        });

        //权限请求
        RxPermissions rxPermission = new RxPermissions(this);
        rxPermission.requestEach(Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE)
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


        /**
         * 通过OpenCV管理Android服务，异步初始化OpenCV
         */
        /*
        BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.i("opencv", "OpenCV loaded successfully");
                        cameraView.enableView();
                        break;
                    default:
                        break;
                }
            }
        };*/
    }
}