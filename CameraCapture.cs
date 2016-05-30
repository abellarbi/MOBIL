

/*
 * MOBIL Binary Descriptor
 * 
 * For any use, please cite :  
 * 
 *  A. Bellarbi, S. Otmane, N. Zenati, and S. Benbelkacem,
 *  “MOBIL: A moments based local binary descriptor,” 
 *  IEEE International Symposium on Mixed and Augmented Reality (ISMAR 2014), 
 *  Munich, Germany.
 * 
 * Author(s): Abdelkader Bellarbi, 
 * CDTA Research Center,
 * Algiers, Algeria.
 * 
 * for any question please email me at : abellarbi@cdta.dz
 * 
 * */


using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu;
using Emgu.Util.TypeEnum;
using Emgu.CV.Flann;
using System.Diagnostics;
using Emgu.CV.Features2D;

using System.IO;
using Emgu.CV.Util;
using System.Globalization;
//using System.Threading;

namespace CameraCapture
{
    public partial class CameraCapture : Form
    {
        MomentsBinaryDescriptor moments_descr;
     
        private Capture _capture;
        private bool _captureInProgress;
        Image<Gray, Byte> img11;
     
            int photoNum = 0;

        public CameraCapture()
        {
            InitializeComponent();
         
            frameWidth = 640;
            frameHeight = 480;

            moments_descr = new MomentsBinaryDescriptor();

        }

        Image<Gray, Byte> frame_gray_input;
        Image<Gray, Byte> frame_gray_model;

        Image<Bgr, Byte> frame_bgr_input;
        Image<Bgr, Byte> frame_bgr_model;

        Image<Bgr, Byte> frame;
        Image<Bgr, Byte> frame_input;
        Image<Gray, Byte> binary_frame;

        Image<Gray, Byte> mask_input;
       

        ORBDetector orb;
        FastDetector fast;

        Image<Bgr, Byte> frame_model;
      
        Image<Gray, Byte> mask_model;

        ImageFeature<Byte>[] img_feature_model;
        ImageFeature<Byte>[] img_feature_input;

        Matrix<Byte> mat_input;
        Matrix<Byte> mat_model;

        VectorOfKeyPoint vector_input;
        VectorOfKeyPoint vector_model; 

        ORBDetector orb_model;
        FastDetector fast_model;

        private void ProcessFrame(object sender, EventArgs arg)
        {
            frame = _capture.QueryFrame();
            frame_bgr_input = frame.Clone();
            frame_gray_input = _capture.QueryGrayFrame();
            DoMainLoop();
        }

        public struct Matched_Point
        {
            public int index_input;
            public int index_model;
            public int dist; 
            
        };

        Matched_Point[] matched_points;

        Image<Bgr, Byte> res_Frame; 
        Image<Bgr, Byte> currentFrameCopy;

        int frameWidth;
        int frameHeight;



        MKeyPoint[] mkpoint_fast = new MKeyPoint[500];
        MKeyPoint[] mkpoint_orb = new MKeyPoint[500];


        MKeyPoint[] mkpoint_fast_model = new MKeyPoint[500];
        MKeyPoint[] mkpoint_orb_model = new MKeyPoint[500];

        PointF[] pts;

        private void DoMainLoop()
        {

             Stopwatch watch = Stopwatch.StartNew();

             moments_descr.frame_gray_input = frame_gray_input;
             moments_descr.frame_bgr_input = frame_bgr_input;

             moments_descr.get_ORB_Points_input();
             long detect_time = watch.ElapsedMilliseconds;
             label4.Text = "Getting keypts : " + detect_time;

             moments_descr.descriptor_input(frame_gray_input);

             long Description_time = watch.ElapsedMilliseconds - detect_time;
             label4.Text = label4.Text + "\nDescription : " + Description_time;

             pts = new PointF[4];
             res_Frame = moments_descr.matching_by_opencv(out pts);
             if (res_Frame != null)
                 pictureBox1.Image = res_Frame.ToBitmap();

             long Matching_time = watch.ElapsedMilliseconds - Description_time;


             label4.Text = label4.Text + "\nMatching : " + Matching_time + "\nTotal : " + watch.ElapsedMilliseconds;
             
             label5.Text = "" + 1000 / watch.ElapsedMilliseconds + "fps";

             frame_bgr_input.DrawPolyline(Array.ConvertAll<PointF, System.Drawing.Point>(pts, System.Drawing.Point.Round), true, new Bgr(Color.Red), 2);

             pictureBox4.Image = frame_bgr_input.Bitmap;

            watch.Stop();
            
        }


        int k = 2;
        double uniquenessThreshold = 0.8;
        Matrix<int> indices;
        Matrix<float> dist;
        Matrix<byte> mask;
        Features2DTracker<Byte>.MatchedImageFeature[] matched_img_feature;// = new Features2DTracker<byte>.MatchedImageFeature[500];
        Features2DTracker<Byte>.MatchedImageFeature[] matched_img_feature_reslt;// = new Features2DTracker<byte>.MatchedImageFeature[500];
        HomographyMatrix homography = null;

        double[] distt;
      
        BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(DistanceType.Hamming);
   
        private void ReleaseData()
        {
            if (_capture != null)
                _capture.Dispose();
        }

    
        private void CameraCapture_Load(object sender, EventArgs e)
        {
            img_feature_model = new ImageFeature<byte>[500];

            img_feature_input = new ImageFeature<byte>[500];

            matched_points = new Matched_Point[500];
        }

        private void button1_Click(object sender, EventArgs e)
        {
            #region if capture is not created, create it now
            if (_capture == null)
            {
                try
                {
                    // http://10.1.30.42/video.cgi?.mjpg

                   _capture = new Capture(comboBox1.SelectedIndex);
                   //  _capture = new Capture(@"D:\Demos\vid-3.avi");
                   //comboBox1.SelectedIndex
                    _capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 640);
                    _capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 480);
                }
                catch (NullReferenceException excpt)
                {
                    MessageBox.Show(excpt.Message);
                }
            }
            #endregion

            if (_capture != null)
            {
                if (_captureInProgress)
                {  //stop the capture
                    button1.Text = "Start Capture";
                    Application.Idle -= ProcessFrame;
                }
                else
                {
                    //start the capture
                    button1.Text = "Stop";
                    Application.Idle += ProcessFrame;
                }

                _captureInProgress = !_captureInProgress;
            }
        }

    
    
     
        private void button3_Click(object sender, EventArgs e)
        {
            if (!System.IO.Directory.Exists("Photos"))
                System.IO.Directory.CreateDirectory("Photos");
            photoNum = 0;
        aaa:
            if (File.Exists("Photos\\image" + photoNum.ToString("000") + ".bmp"))
            {
                photoNum++;
                goto aaa;
            }
            else
            {
               // Bitmap bmp = new Bitmap(Width+10, Height+40);
               // this.DrawToBitmap(bmp, Bounds);

              //  bmp.Save("Photos\\image" + photoNum.ToString("000") + ".bmp");
                
                frame.Bitmap.Save("Photos\\image" + photoNum.ToString("000") + ".bmp");

                pictureBox2.Image = frame.Bitmap;

                frame_gray_model = new Image<Gray, byte>(frame.Bitmap);
                 
                moments_descr.frame_gray_model = frame_gray_model;

                moments_descr.frame_bgr_model = new Image<Bgr, byte>(frame.Bitmap); ;
             
                moments_descr.get_ORB_Points_model();

                moments_descr.descriptor_model(frame_gray_model);
            }


        }

        bool do_match = true;

        private void button2_Click(object sender, EventArgs e)
        {
            do_match = !do_match;
        }





        private void button4_Click(object sender, EventArgs e)
        {
            //openFileDialog1.InitialDirectory = Application.StartupPath;

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {

                

                frame_bgr_model = new Image<Bgr, byte>(openFileDialog1.FileName);

                pictureBox2.Image = frame_bgr_model.Bitmap;

                frame_gray_model = new Image<Gray, byte>(openFileDialog1.FileName);


                moments_descr.frame_bgr_model = frame_bgr_model;

                moments_descr.frame_gray_model = frame_gray_model;

                moments_descr.get_ORB_Points_model();

                moments_descr.descriptor_model(frame_gray_model);
                
            }
        }

        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {

        }

        private void button5_Click(object sender, EventArgs e)
        {
            //openFileDialog1.InitialDirectory = Application.StartupPath;

            if (openFileDialog2.ShowDialog() == DialogResult.OK)
            {



                frame_bgr_input = new Image<Bgr, byte>(openFileDialog2.FileName);

                pictureBox4.Image = frame_bgr_input.Bitmap;

                frame_gray_input = new Image<Gray, byte>(openFileDialog2.FileName);


                moments_descr.frame_bgr_input = frame_bgr_input;

                moments_descr.frame_gray_input = frame_gray_input;

              //  moments_descr.get_ORB_Points_input();

              //  moments_descr.descriptor_input(frame_gray_input);

                DoMainLoop();

            }
        }

        private void openFileDialog2_FileOk(object sender, CancelEventArgs e)
        {

        }

        private void button6_Click(object sender, EventArgs e)
        {
           // openFileDialog1.InitialDirectory = Application.StartupPath;

            if (openFileDialog3.ShowDialog() == DialogResult.OK)
            {

                string[] mat_rows = new string[3];
                mat_rows = File.ReadAllLines(openFileDialog3.FileName);

                string[][] mat_cells = new string[3][];

             //   mat_cells[0] = new string[8];
             //   mat_cells[1] = new string[8];
             //   mat_cells[2] = new string[8];

                mat_cells[0] = mat_rows[0].Split(new char[] { ' ' });
                mat_cells[1] = mat_rows[1].Split(new char[] { ' ' });
                mat_cells[2] = mat_rows[2].Split(new char[] { ' ' });


                Matrix<double> mat_homography = new Matrix<double>(new Size(3, 3));

             
                int k = 0;
                for (int i = 0; i < 3; i++)
                {
                    k = 0;
                    for (int j = 0; j < mat_cells[i].Length; j++)
                    {
                        string aaa = "";
                        aaa = mat_cells[i][j];

                      //  char[] bbb = aaa.ToCharArray();
                       
                      //  aaa.Trim(new char[] { ' ', '\n', '\t' });
                    
                        if (aaa != "" && aaa != " ")
                            mat_homography[i, k++] = double.Parse(aaa, CultureInfo.InvariantCulture);

                    }

                   // pictureBox2.Image = frame_bgr_model.Bitmap;
                }

                double[,] test_H = new double[moments_descr.matchedFeatures.Length, 3];


                test_H = moments_descr.verify_homography(mat_homography, 280, 3f);

                float num_matched = 0;

                float num_true_matched = 0;

                for (int i = 0; i < moments_descr.matchedFeatures.Length; i++)
                {

                    if (test_H[i, 1] == 1)
                    {
                        num_matched++;
                        if (test_H[i, 2] == 1)
                            num_true_matched++;


                    }
                }


                // watch.Stop();
                label6.Text = " all detected points  : " + moments_descr.points_number_octave0_input +
                              "\n All matched : " + moments_descr.matchedFeatures.Length +
                              "\n with <20 : " + num_matched +
                              "\n exact : " + num_true_matched +
                                "\n\n All matched : " + num_matched + " / " + moments_descr.points_number_octave0_input + " : " + num_matched / moments_descr.points_number_octave0_input +
                               "\n exact (recall) : " + num_true_matched + " / " + moments_descr.points_number_octave0_input + " : " + num_true_matched / moments_descr.points_number_octave0_input +
                               "\n Precsion : " + num_true_matched + " / " + num_matched + " : " + num_true_matched / num_matched;
                              
         

            }
        }

       
    }
}
