
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
using System.Linq;
using System.Text;

using System.Drawing;
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
using System.Threading;



namespace CameraCapture
{
    class MomentsBinaryDescriptor
    { 

        public float scale_factor = 1.2f;
        static int max_kpoints = 500;
        public Image<Gray, Byte> frame_gray_model;
        public Image<Gray, Byte> frame_gray_input;
       
        public Image<Bgr, Byte> frame_bgr_model;
        public Image<Bgr, Byte> frame_bgr_input;

        public ORBDetector orb_model;
        public ORBDetector orb_input; 

        public MKeyPoint[] mkpoint_orb_model;
        public MKeyPoint[] mkpoint_orb_input; 

        public Image<Gray, Byte>[] image_model_scaled = new Image<Gray, byte>[8]; 

        public int patch_size = 8;
        public int mat_width = 12;
        public int points_number_octave0_input;
        public int points_number_octave0_model;

        public int n_bits = 56;
        public int n_bits_moments = 5;

        public int r =12;
        
        public int angl = 12;

        public BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(DistanceType.Hamming2);

        public int k = 2;
        public double uniquenessThreshold = 0.8;
        public Matrix<int> indices;
        public Matrix<float> dist;
        public Matrix<byte> mask;
        public Features2DTracker<Byte>.MatchedImageFeature[] matched_img_feature;// = new Features2DTracker<byte>.MatchedImageFeature[500];
        public Features2DTracker<Byte>.MatchedImageFeature[] matched_img_feature_reslt;// = new Features2DTracker<byte>.MatchedImageFeature[500];
        public HomographyMatrix homography = null;

        public Matrix<Byte> mat_input;
        public Matrix<Byte> mat_model;
        
        public bool[,] mat_input_t;
        public bool[,] mat_model_t;

        public VectorOfKeyPoint vector_input;
        public VectorOfKeyPoint vector_model; 

        public VectorOfKeyPoint vector_model_scaled;


        ImageFeature<byte>[] modelFeatures = new ImageFeature<byte>[max_kpoints];

        ImageFeature<byte>[] inputFeatures = new ImageFeature<byte>[max_kpoints];

        //ThresholdBinary(new Gray(threshold_value), new Gray(255));

        //public int threshold_value_model = 100;
        //public int threshold_value_input = 100; 

          
            //frame_gray_input = frame_gray_input.SmoothBlur(3, 3);
            //frame_gray_input = frame_gray_input.Dilate(1);
            //frame_gray_input = frame_gray_input.Erode(1);

            //int  otsu_threshold = calculateThreshold(frame_gray_input);
             

            //frame_gray_input = frame_gray_input.ThresholdBinary(new Gray(otsu_threshold), new Gray(255));





        public struct Point_Descriptor_Model
        {
            public PointF pt;

            public float scale;

            public bool[] binary_discriptor;

            public bool[] binary_discriptor2;

            public int[] gradiant_discriptor; 

            public Image<Gray, Byte> sub_frame_gray;
            public Image<Gray, Byte> sub_rotated_gray;

            public Image<Gray, Byte> sub_polar_gray;

            public double angle;
        };

        public Point_Descriptor_Model[] point_Descriptor_Model;

        public struct Point_Descriptor_Input
        {
            public PointF pt;

            public float scale;

            public bool[] binary_discriptor;

            public bool[] binary_discriptor2; 

            public int[] gradiant_discriptor; 

            public Image<Gray, Byte> sub_frame_gray;
            public Image<Gray, Byte> sub_rotated_gray;


            public Image<Gray, Byte> sub_polar_gray;

            public double angle;

        };

        public Point_Descriptor_Input[] point_Descriptor_Input;

        public struct Matched_Point
        {
            public PointF point_input;
            public PointF point_model;

            public int index_input;
            public int index_model;

            public int[,] tab_dist;
            public int dist;
            public int dist2;

            public int min_dist;  

            int dist_hamming;
        };

        public Matched_Point[] matched_points;


        public void get_ORB_Points_model()
        {

            orb_model = new ORBDetector(max_kpoints);

            mkpoint_orb_model = orb_model.DetectKeyPoints(frame_gray_model, null);
           
           modelFeatures = new ImageFeature<byte>[mkpoint_orb_model.Length];


            mkpoint_orb_model = orb_model.DetectKeyPoints(frame_gray_model, null);
            for (int i = 0; i < mkpoint_orb_model.Length; i++)
            {
               /* if (mkpoint_orb_model[i].Octave > 0)
                {
                    points_number_octave0_model = i;
                    break;
                }*/
                modelFeatures[i] = new ImageFeature<byte>();
                modelFeatures[i].KeyPoint = mkpoint_orb_model[i];
            }
        }

        public void get_ORB_Points_input()
        {

            orb_input = new ORBDetector(max_kpoints);
            points_number_octave0_input = 0;
            mkpoint_orb_input = orb_input.DetectKeyPoints(frame_gray_input, null);
            if (mkpoint_orb_input.Length > 4)
            {
                int j = 0;
                while (mkpoint_orb_input[j].Octave == 0)
                {
                    j++;
                    points_number_octave0_input = j;
                }

                // points_number_octave0_input = (int)(max_kpoints * 0.219);
                inputFeatures = new ImageFeature<byte>[points_number_octave0_input];


                for (int i = 0; i < points_number_octave0_input; i++)
                {

                    /*  if (mkpoint_orb_input[i].Octave > 0)
                      {
                          points_number_octave0_input = i;
                          break;
                      }*/
                  

                    inputFeatures[i] = new ImageFeature<byte>();
                    inputFeatures[i].KeyPoint = mkpoint_orb_input[i];
                }
            }
        }

        public Image<Gray, Byte> sub_image_cerle(Image<Gray, Byte> image_gray, PointF p, int r)
        {

            Image<Gray, Byte> patch_gray = new Image<Gray, byte>(r * 2, r * 2);

            byte[, ,] data_cercle = new byte[r * 2, r * 2, 1];

            int ii = 0;
            int jj = 0;
            if (p.X + r < image_gray.Width && p.X - r > 0 && p.Y + r < image_gray.Height && p.Y - r > 0)
                for (int i = 0; i < r * 2; i++)
                    for (int j = 0; j < r * 2; j++)
                    {
                        if ((i - r) * (i - r) + (j - r) * (j - r) <= r * r)
                            data_cercle[i, j, 0] = image_gray.Data[i + (int)p.Y - r, j + (int)p.X - r, 0];
                        else
                            data_cercle[i, j, 0] = 0;
                    }

            patch_gray.Data = data_cercle;

            return patch_gray;

        }

        public Image<Gray, Byte> sub_image_square(Image<Gray, Byte> image_gray, PointF p, int r)
        {

            Image<Gray, Byte> patch_gray = new Image<Gray, byte>(r * 2, r * 2);

            byte[, ,] data_square = new byte[r * 2, r * 2, 1];

            int ii = 0;
            int jj = 0;
            if (p.X + r < image_gray.Width && p.X - r > 0 && p.Y + r < image_gray.Height && p.Y - r > 0)
            for (int i = 0; i < r * 2; i++)
                for (int j = 0; j < r * 2; j++)
                {
                       data_square[i, j, 0] = image_gray.Data[i + (int)p.Y - r, j + (int)p.X - r, 0];
                }

            patch_gray.Data = data_square;

            return patch_gray;


        }

        public void descriptor_input(Image<Gray, Byte> frame_gray_input) 
        {
            point_Descriptor_Input = new Point_Descriptor_Input[points_number_octave0_input];

            get_descriptor_input(frame_gray_input);

        }



        public void descriptor_model(Image<Gray, Byte> frame_gray_model) 
        {
            point_Descriptor_Model = new Point_Descriptor_Model[max_kpoints];

            get_descriptor_model(frame_gray_model);

            //ThresholdBinary(new Gray(threshold_value), new Gray(255));

        }

        public void get_descriptor_input(Image<Gray, Byte> frame_gray_input)
        {

//
         
           frame_gray_input = frame_gray_input.SmoothBlur(1,1);
          //  frame_gray_input = frame_gray_input.Dilate(1);
          //  frame_gray_input = frame_gray_input.Erode(1);

          
          //  otsu_threshold = calculateThreshold(frame_gray_input);


            if (mkpoint_orb_input.Length > 4)
            {

                //  frame_gray_input = frame_gray_input.ThresholdBinary(new Gray(otsu_threshold), new Gray(255));

                mat_input = new Matrix<Byte>(n_bits * n_bits_moments, mkpoint_orb_input.Length);

                // frame_gray_input = frame_gray_input.Canny(new Gray(threshold_value), new Gray(threshold_value)); 

                for (int i = 0; i < mkpoint_orb_input.Length; i++)
                {

                    if (mkpoint_orb_input[i].Octave > 0)
                    {
                        points_number_octave0_input = i;
                        break;
                    }

                    point_Descriptor_Input[i].binary_discriptor2 = new bool[n_bits * n_bits_moments];
                    point_Descriptor_Input[i].binary_discriptor2 = get_patch_description_and_rotation_by_moments(frame_gray_input, mkpoint_orb_input[i].Point, r, out point_Descriptor_Input[i].sub_frame_gray, out point_Descriptor_Input[i].angle);
                    point_Descriptor_Input[i].pt = mkpoint_orb_input[i].Point;

                    inputFeatures[i].Descriptor = new byte[n_bits * n_bits_moments / 8];

                    inputFeatures[i].Descriptor = bool_to_byte(point_Descriptor_Input[i].binary_discriptor2, n_bits * n_bits_moments);

                    //   point_Descriptor_Input[i].sub_polar_gray = get_polar_image(frame_gray_input, mkpoint_orb_input[i].Point, r, angl, out point_Descriptor_Input[i].sub_frame_gray, out  point_Descriptor_Input[i].angle);
                    //   point_Descriptor_Input[i].sub_rotated_gray = get_rotated_image(frame_gray_input, mkpoint_orb_input[i].Point, r, angl, out point_Descriptor_Input[i].sub_frame_gray, out  point_Descriptor_Input[i].angle);

                }

            }
        }

        public void get_descriptor_model(Image<Gray, Byte> frame_gray_model)
        {

           // frame_gray_model = frame_gray_model.SmoothBlur(1, 1);
            mat_model_t = new bool[n_bits * n_bits_moments, mkpoint_orb_model.Length];

            for (int j = 0; j < 8; j++)
            {

                image_model_scaled[j] = frame_gray_model.Resize(1 / Math.Pow(scale_factor, j), Emgu.CV.CvEnum.INTER.CV_INTER_LINEAR);

                for (int i = 0; i < mkpoint_orb_model.Length; i++)
                {

                    /* if (mkpoint_orb_model[i].Octave > 0)
                     {
                         points_number_octave0_model = i;
                         break;
                     }*/


                    if (mkpoint_orb_model[i].Octave == j)
                    {

                        PointF ptf = new PointF((float)(mkpoint_orb_model[i].Point.X * 1 / Math.Pow(scale_factor, j)), (float)(mkpoint_orb_model[i].Point.Y * 1 / Math.Pow(scale_factor, j)));

                        point_Descriptor_Model[i].binary_discriptor2 = new bool[n_bits * n_bits_moments];
                        point_Descriptor_Model[i].binary_discriptor2 = get_patch_description_and_rotation_by_moments(image_model_scaled[j], ptf, r, out point_Descriptor_Model[i].sub_frame_gray, out  point_Descriptor_Model[i].angle);

                        point_Descriptor_Model[i].pt = mkpoint_orb_model[i].Point;

                        point_Descriptor_Model[i].scale = mkpoint_orb_model[i].Octave;
                        modelFeatures[i].Descriptor = new byte[n_bits * n_bits_moments / 8];

                        modelFeatures[i].Descriptor = bool_to_byte(point_Descriptor_Model[i].binary_discriptor2, n_bits * n_bits_moments);

                    }
                }
            }

            //   points_number_octave0_model = mkpoint_orb_model.Length;


        }


     
        double[] tab_input;


       

        public bool[] get_patch_description_and_rotation_by_moments(Image<Gray, byte> frame_gray, PointF pointF, int r, out Image<Gray, byte> sub_frame_gray, out double angle)
        {

            float r2 = r *1.5f;

            Image<Gray, byte> sub_frame_gray_cercle = sub_image_cerle(frame_gray, pointF, (int)(r2));

           // Image<Gray, byte> sub_frame_gray_cercle2 = sub_image_cerle(frame_gray, pointF, (int)(r));
            MCvMoments moments = sub_frame_gray_cercle.GetMoments(false);

            double m00 = moments.m00;
            double m01 = moments.m01;
            double m10 = moments.m10;

            double center_x = m10 / m00;
            double center_y = m01 / m00;

            angle = Math.Atan2((r2) - center_y, (r2) - center_x);
            double deg_ang1 = Emgu.CV.Geodetic.GeodeticCoordinate.RadianToDegree(angle);

            sub_frame_gray_cercle = sub_frame_gray_cercle.Rotate(-deg_ang1, new Gray(0), true);

            sub_frame_gray = sub_frame_gray_cercle.GetSubRect(new Rectangle(new Point(r / 2, r / 2), new Size(r * 2, r * 2)));
          

          //  sub_frame_gray = sub_image_square(sub_frame_gray_cercle, new PointF((r2), (r2)), r);


            bool[] compare_moments = new bool[n_bits* n_bits_moments];

            compare_moments = compare_sub_patch_moments5(sub_frame_gray);

            

            return compare_moments;

        }

     



        public bool[] compare_sub_patch_moments5(Image<Gray, byte> patch_gray)
        {
            bool[] compare_moments = new bool[n_bits * n_bits_moments];

            double[,] patch_moments = new double[16, n_bits_moments];

            int ii = 0;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                {

                    //   Image<Gray, byte> sub_patch_gray = patch_gray.GetSubRect(new Rectangle(r / 2 * i, r / 2 * j, r / 2, r / 2));
                    //   MCvMoments moments = sub_patch_gray.GetMoments(false);

                    //MCvMoments moments = patch_gray.GetSubRect(new Rectangle(r / 2 * i, r / 2 * j, r / 2, r / 2)).GetMoments(false);

               
                    patch_gray.ROI = new Rectangle(r / 2 * i, r / 2 * j, r / 2, r / 2);



                    MCvMoments moments = patch_gray.GetMoments(false);


                    
                    patch_gray.ROI = Rectangle.Empty;



                    /*
                    patch_moments[ii, 2] = hu_moments.hu4 / hu_moments.hu1;

                    patch_moments[ii, 4] = (hu_moments.hu2 / hu_moments.hu1);
                    patch_moments[ii, 3] = (hu_moments.hu3 / hu_moments.hu1); //2 * moments.m11 / (moments.m20 - moments.m02);

                    patch_moments[ii, 0] = hu_moments.hu7 / hu_moments.hu1;// / hu_moments.m00;
                    patch_moments[ii, 1] = hu_moments.hu6 / hu_moments.hu1;// / hu_moments.m00;
                    */
                 
                    
                 //   patch_moments[ii, 2] = moments.m11 / moments.m00;
                    patch_moments[ii, 0] = moments.m00;
                   patch_moments[ii, 1] = moments.m20;
                   patch_moments[ii, 2] = moments.m02;
                    patch_moments[ii, 3] = moments.m10 / moments.m00;
                    patch_moments[ii, 4] = moments.m01 / moments.m00;


                //    patch_moments[ii, 1] =moments.m00;
               //      patch_moments[ii, 2] = moments.m00;
               //     patch_moments[ii, 3] = moments.m00;
               //     patch_moments[ii, 4] =moments.m00;


                    /*
                       patch_moments[ii, 0] = moments.m01 / moments.m00;
                     patch_moments[ii, 1] = moments.m10 / moments.m00;

                 //   patch_moments[ii, 4] = moments.m10 / moments.m01;
               //     patch_moments[ii, 3] = moments.m10 / moments.m01; //2 * moments.m11 / (moments.m20 - moments.m02);

                    


                 //   patch_moments[ii, 4] = moments.m02 / moments.m00 - (moments.m01 * moments.m01) / (moments.m00 * moments.m00);
                 //   patch_moments[ii, 3] = moments.m20 / moments.m00 - (moments.m10 * moments.m10) / (moments.m00 * moments.m00);


                

                    //  patch_moments[ii, 5] = moments.m11;// / moments.m00;

                    //  patch_moments[ii, 0] = moments.m01 / moments.m00;
                    //   patch_moments[ii, 1] = moments.m10 / moments.m00;

                   patch_moments[ii, 3] = moments.m02;// / moments.m00;
                     patch_moments[ii, 4] = moments.m20;// / moments.m00;

                    //  patch_moments[ii, 4] = moments.m02 / moments.m00;
                    //  patch_moments[ii, 5] = moments.m20 / moments.m00;
                    */

                    ii++;
                }


            for (int i = 0; i < n_bits_moments; i++)
            {

                //////////////////////////////////////
                // square 5 with neabours 

                if (patch_moments[5, i] >= patch_moments[0, i]) 
                    compare_moments[0+n_bits*i] = true; 
                else 
                    compare_moments[0+n_bits*i] = false;





                if (patch_moments[5, i] >= patch_moments[2, i]) compare_moments[1+n_bits*i] = true; else compare_moments[1+n_bits*i] = false;
                if (patch_moments[5, i] >= patch_moments[8, i]) compare_moments[2+n_bits*i] = true; else compare_moments[2+n_bits*i] = false;
                if (patch_moments[5, i] >= patch_moments[10, i]) compare_moments[3+n_bits*i] = true; else compare_moments[3+n_bits*i] = false;
            
                //////////////////////////////////////
                // square 6 with neabours 

                if (patch_moments[6, i] >= patch_moments[1, i]) compare_moments[4+n_bits*i] = true; else compare_moments[4+n_bits*i] = false;
                if (patch_moments[6, i] >= patch_moments[3, i]) compare_moments[5+n_bits*i] = true; else compare_moments[5+n_bits*i] = false;
                if (patch_moments[6, i] >= patch_moments[9, i]) compare_moments[6+n_bits*i] = true; else compare_moments[6+n_bits*i] = false;
                if (patch_moments[6, i] >= patch_moments[11, i]) compare_moments[7+n_bits*i] = true; else compare_moments[7+n_bits*i] = false;


                //////////////////////////////////////
                // square 9 with neabours 

                if (patch_moments[9, i] >= patch_moments[4, i]) compare_moments[8+n_bits*i] = true; else compare_moments[8+n_bits*i] = false;
                if (patch_moments[9, i] >= patch_moments[12, i]) compare_moments[9+n_bits*i] = true; else compare_moments[9+n_bits*i] = false;
                if (patch_moments[9, i] >= patch_moments[14, i]) compare_moments[10+n_bits*i] = true; else compare_moments[10+n_bits*i] = false;

                // square 10 with neabours 


                if (patch_moments[10, i] >= patch_moments[7, i]) compare_moments[11+n_bits*i] = true; else compare_moments[11+n_bits*i] = false;
                if (patch_moments[10, i] >= patch_moments[13, i]) compare_moments[12+n_bits*i] = true; else compare_moments[12+n_bits*i] = false;
                if (patch_moments[10, i] >= patch_moments[15, i]) compare_moments[13+n_bits*i] = true; else compare_moments[13+n_bits*i] = false;


                // square 5 with neabours 


                if (patch_moments[5, i] >= patch_moments[7, i]) compare_moments[14+n_bits*i] = true; else compare_moments[14+n_bits*i] = false;
                if (patch_moments[5, i] >= patch_moments[13, i]) compare_moments[15+n_bits*i] = true; else compare_moments[15+n_bits*i] = false;
                if (patch_moments[5, i] >= patch_moments[15, i]) compare_moments[16+n_bits*i] = true; else compare_moments[16+n_bits*i] = false;

                // square 6 with neabours 


                if (patch_moments[6, i] >= patch_moments[4, i]) compare_moments[17+n_bits*i] = true; else compare_moments[17+n_bits*i] = false;
                if (patch_moments[6, i] >= patch_moments[12, i]) compare_moments[18+n_bits*i] = true; else compare_moments[18+n_bits*i] = false;
                if (patch_moments[6, i] >= patch_moments[14, i]) compare_moments[19+n_bits*i] = true; else compare_moments[19+n_bits*i] = false;

                // square 9 with neabours 


                if (patch_moments[9, i] >= patch_moments[1, i]) compare_moments[20+n_bits*i] = true; else compare_moments[20+n_bits*i] = false;
                if (patch_moments[9, i] >= patch_moments[3, i]) compare_moments[21+n_bits*i] = true; else compare_moments[21+n_bits*i] = false;
                if (patch_moments[9, i] >= patch_moments[11, i]) compare_moments[22+n_bits*i] = true; else compare_moments[22+n_bits*i] = false;

                // square 10 with neabours 


                if (patch_moments[10, i] >= patch_moments[0, i]) compare_moments[23 + n_bits * i] = true; else compare_moments[23 + n_bits * i] = false;
                if (patch_moments[10, i] >= patch_moments[2, i]) compare_moments[24 + n_bits * i] = true; else compare_moments[24 + n_bits * i] = false;
                if (patch_moments[10, i] >= patch_moments[8, i]) compare_moments[25 + n_bits * i] = true; else compare_moments[25 + n_bits * i] = false;

                // square 0,3,12,15 with neabours 


                if (patch_moments[0, i] >= patch_moments[3, i]) compare_moments[26 + n_bits * i] = true; else compare_moments[26 + n_bits * i] = false;
                if (patch_moments[0, i] >= patch_moments[12, i]) compare_moments[27 + n_bits * i] = true; else compare_moments[27 + n_bits * i] = false;
                if (patch_moments[0, i] >= patch_moments[15, i]) compare_moments[28 + n_bits * i] = true; else compare_moments[28 + n_bits * i] = false;
                if (patch_moments[3, i] >= patch_moments[12, i]) compare_moments[29 + n_bits * i] = true; else compare_moments[29 + n_bits * i] = false;
                if (patch_moments[3, i] >= patch_moments[15, i]) compare_moments[30 + n_bits * i] = true; else compare_moments[30 + n_bits * i] = false;
                if (patch_moments[12, i] >= patch_moments[15, i]) compare_moments[31 + n_bits * i] = true; else compare_moments[31 + n_bits * i] = false;



                // square 1,2,4,8 with neabours 


                if (patch_moments[1, i] >= patch_moments[14, i]) compare_moments[32 + n_bits * i] = true; else compare_moments[32 + n_bits * i] = false;
                if (patch_moments[2, i] >= patch_moments[13, i]) compare_moments[33 + n_bits * i] = true; else compare_moments[33 + n_bits * i] = false;
                if (patch_moments[4, i] >= patch_moments[11, i]) compare_moments[34 + n_bits * i] = true; else compare_moments[34 + n_bits * i] = false;
                if (patch_moments[8, i] >= patch_moments[7, i]) compare_moments[35 + n_bits * i] = true; else compare_moments[35 + n_bits * i] = false;


                // square 1,2,4,8 with neabours 


                if (patch_moments[0, i] >= patch_moments[1, i]) compare_moments[36 + n_bits * i] = true; else compare_moments[36 + n_bits * i] = false;
                if (patch_moments[0, i] >= patch_moments[4, i]) compare_moments[37 + n_bits * i] = true; else compare_moments[37 + n_bits * i] = false;
                if (patch_moments[3, i] >= patch_moments[2, i]) compare_moments[39 + n_bits * i] = true; else compare_moments[39 + n_bits * i] = false;
                if (patch_moments[3, i] >= patch_moments[7, i]) compare_moments[40 + n_bits * i] = true; else compare_moments[40 + n_bits * i] = false;
                if (patch_moments[12, i] >= patch_moments[8, i]) compare_moments[41 + n_bits * i] = true; else compare_moments[41 + n_bits * i] = false;
                if (patch_moments[12, i] >= patch_moments[13, i]) compare_moments[42 + n_bits * i] = true; else compare_moments[42 + n_bits * i] = false;
                if (patch_moments[15, i] >= patch_moments[14, i]) compare_moments[43 + n_bits * i] = true; else compare_moments[43 + n_bits * i] = false;
                if (patch_moments[15, i] >= patch_moments[11, i]) compare_moments[44 + n_bits * i] = true; else compare_moments[44 + n_bits * i] = false;


                // square 2,4,11 with neabours 


                if (patch_moments[2, i] >= patch_moments[11, i]) compare_moments[45 + n_bits * i] = true; else compare_moments[45 + n_bits * i] = false;
                if (patch_moments[11, i] >= patch_moments[13, i]) compare_moments[46 + n_bits * i] = true; else compare_moments[46 + n_bits * i] = false;
                if (patch_moments[13, i] >= patch_moments[4, i]) compare_moments[47 + n_bits * i] = true; else compare_moments[47 + n_bits * i] = false;
                if (patch_moments[4, i] >= patch_moments[2, i]) compare_moments[48 + n_bits * i] = true; else compare_moments[48 + n_bits * i] = false;



                // square 1,2,4,8 with neabours 


                if (patch_moments[1, i] >= patch_moments[7, i]) compare_moments[49 + n_bits * i] = true; else compare_moments[49 + n_bits * i] = false;
                if (patch_moments[7, i] >= patch_moments[14, i]) compare_moments[50 + n_bits * i] = true; else compare_moments[50 + n_bits * i] = false;
                if (patch_moments[14, i] >= patch_moments[8, i]) compare_moments[51 + n_bits * i] = true; else compare_moments[51 + n_bits * i] = false;
                if (patch_moments[8, i] >= patch_moments[1, i]) compare_moments[52 + n_bits * i] = true; else compare_moments[52 + n_bits * i] = false;
                if (patch_moments[1, i] >= patch_moments[10, i]) compare_moments[53 + n_bits * i] = true; else compare_moments[53 + n_bits * i] = false;
                if (patch_moments[7, i] >= patch_moments[9, i]) compare_moments[54 + n_bits * i] = true; else compare_moments[54 + n_bits * i] = false;
                if (patch_moments[14, i] >= patch_moments[5, i]) compare_moments[55 + n_bits * i] = true; else compare_moments[55 + n_bits * i] = false;
               
            }

            return compare_moments;
        }



        public Features2DTracker<Byte>.MatchedImageFeature[] matchedFeatures;
   
        public Image<Bgr, Byte> matching_by_opencv(out PointF[] pts)
        {

            //  Image<Gray, Byte> modelImage = frame_gray_model;
            //  Image<Gray, Byte> observedImage = frame_gray_input;
            // Extract features from the object image
            pts = new PointF[4];

            if ( points_number_octave0_input > 4) 
            {
                Features2DTracker<Byte> tracker;
                // Create a SURF Tracker using k-d Tree
                tracker = new Features2DTracker<Byte>(modelFeatures);


                matchedFeatures = tracker.MatchFeature(inputFeatures, 1);
                matchedFeatures = Features2DTracker<Byte>.VoteForUniqueness(matchedFeatures, 0.9);

               // matchedFeatures = Features2DTracker<Byte>.VoteForSizeAndOrientation(matchedFeatures, 1.2, 180);

                HomographyMatrix homography = Features2DTracker<Byte>.GetHomographyMatrixFromMatchedFeatures(matchedFeatures);
                
                // Merge the object image and the observed image into one image for display
                Image<Bgr, Byte> res = frame_bgr_model.ConcateHorizontal(frame_bgr_input);


              double[,] test_H = new double[matchedFeatures.Length, 3];
              float num_matched = 0;

              float num_true_matched = 0;

              if (homography != null)
              {
                  test_H = verify_homography(homography, 280, 2f);



                  for (int i = 0; i < matchedFeatures.Length; i++)
                  {

                      if (test_H[i, 1] == 1)
                      {
                          num_matched++;
                          if (test_H[i, 2] == 1)
                          {
                              num_true_matched++;
                              PointF p = matchedFeatures[i].ObservedFeature.KeyPoint.Point;
                              p.X += frame_bgr_model.Width;
                              res.Draw(new LineSegment2DF(matchedFeatures[i].SimilarFeatures[0].Feature.KeyPoint.Point, p), new Bgr(Color.Green), 2);
                          }
                          else
                          {
                              PointF p = matchedFeatures[i].ObservedFeature.KeyPoint.Point;
                              p.X += frame_bgr_model.Width;
                              //res.Draw(new LineSegment2DF(matchedFeatures[i].SimilarFeatures[0].Feature.KeyPoint.Point, p), new Bgr(Color.Red), 2);
                          }

                      }
                  }


                  // draw a rectangle along the projected model
                  Rectangle rect = frame_bgr_model.ROI;
                  pts = new PointF[] { 
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top) };

                  homography.ProjectPoints(pts);
                  PointF[] pts1 = new PointF[4];
                  for (int i = 0; i < pts.Length; i++)
                  {
                      pts1[i].Y += pts[i].Y ;
                      pts1[i].X += pts[i].X + frame_bgr_model.Width;
                  }
                // res.DrawPolyline(Array.ConvertAll<PointF, System.Drawing.Point>(pts1, System.Drawing.Point.Round), true, new Bgr(Color.Red), 4);





              }

              

             

                return res;


            }

            return null;
        }

      
        public byte[] bool_to_byte(bool[] arr, int n)
        {
            bool[][] mat = new bool[n / 8][];
            byte[] mat_byte = new byte[n / 8];
            for (int i = 0; i < n / 8; i++)
            {
                mat[i] = new bool[8];
                for (int j = 0; j < 8; j++)
                    mat[i][j] = arr[i * 8 + j];
            }

            for (int i = 0; i < n / 8; i++)
            {
                mat_byte[i] = 0;
                foreach (bool b in mat[i])
                {
                    mat_byte[i] <<= 1;
                    if (b) mat_byte[i] |= 1;
                }
            }

            return mat_byte;
        }


        public double[,] verify_homography(Matrix<double> H, int dist_min, double epsilon)
        {
            double[,] test_homography = new double[matchedFeatures.Length, 3];

            for (int i = 0; i < matchedFeatures.Length; i++)
            {

                Matrix<double> vector_output = new Matrix<double>(1, 3);
                Matrix<double> vector_model = new Matrix<double>(1, 3);

                vector_model[0, 0] = matchedFeatures[i].SimilarFeatures[0].Feature.KeyPoint.Point.X;
                vector_model[0, 1] = matchedFeatures[i].SimilarFeatures[0].Feature.KeyPoint.Point.Y;
                vector_model[0, 2] = 1;

                vector_model = vector_model.Transpose();

                vector_output = H * vector_model;

                vector_output = vector_output / vector_output[2,0];

                if (matchedFeatures[i].SimilarFeatures[0].Distance <= dist_min)
                    test_homography[i, 1] = 1;
                else
                    test_homography[i, 1] = 0;

                test_homography[i, 0] = Math.Sqrt(Math.Pow(vector_output[0, 0] - matchedFeatures[i].ObservedFeature.KeyPoint.Point.X, 2) + Math.Pow(vector_output[1, 0] - matchedFeatures[i].ObservedFeature.KeyPoint.Point.Y, 2));

                if (test_homography[i, 0] <= epsilon)
                    test_homography[i, 2] = 1;
                else
                    test_homography[i, 2] = 0;


            }

            return test_homography;
        }

        public double[] diffr(double[] tab1, double[] tab2)
        {
            double[] diff = new double[7];

            for (int i = 0; i < 7; i++)
                diff[i] = Math.Abs(tab1[i] - tab2[i]);

            return diff;
        }

        public int diff_binary(byte[, ,] tab1, byte[, ,] tab2, int w, int h)
        {

            int dist_hamming = 0;

            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    if (tab1[j, i, 0] != tab2[j, i, 0])
                        dist_hamming++;
                }

            return dist_hamming;

        }
        public int diff_binary(bool[] tab1, bool[] tab2, int n)
        {

            int dist_hamming = 0;

            for (int i = 0; i < n; i++)
            {
                if (tab1[i] != tab2[i])
                    dist_hamming++;
            }

            return dist_hamming;

        }

        public int diff_binary2(bool[,] tab1, bool[,] tab2, int n)
        {

            int dist_hamming = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n_bits_moments; j++)
                {
                    if (tab1[i, j] != tab2[i, j])
                        dist_hamming++;
                    // else
                    //     dist_hamming--;
                }

                /*
                if (n == 4 && dist_hamming >= 10)
                {
                    dist_hamming = n * n_bits_moments;
                    break;
                }
                */
            }

            return dist_hamming;

        }

        public int diff_binary_by_portion(byte[, ,] tab1, byte[, ,] tab2, int w, int h)
        {

            int dist_hamming = 0;

            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    if (tab1[j, i, 0] != tab2[j, i, 0])
                        dist_hamming++;
                }

            return dist_hamming;

        }


        public double[] diffr_ponderee(double[] tab1, double[] tab2)
        {

            double[] diff = new double[8];

            double sum = 0;

            for (int i = 0; i < 7; i++)
            {
                diff[i] = Math.Abs((tab1[i] - tab2[i]) / (Math.Max(tab1[i], tab2[i])));
                sum = sum + diff[i];
            }

            diff[7] = sum - diff[4] -diff[5];

            return diff;

        }

        public double[] diffr_ponderee1(double[] tab1, double[] tab2)
        {
            double[] diff = new double[8];

            double sum = 0;

            for (int i = 0; i < 7; i++)
            {
                diff[i] = Math.Abs((tab1[i] - tab2[i]) / (Math.Max(tab1[i], tab2[i])));
                sum = sum + diff[i];
            }

            diff[7] = sum - diff[4] - diff[5];

            return diff;
        }

   
        public Matched_Point match_points_with_Coeff(Point_Descriptor_Input point_input, Point_Descriptor_Model[] points_models)
        {

            Matched_Point matched_point = new Matched_Point();
            int min_distance = 0;
            int index = 0;
            
            int min_distance2 = 0;
            int index2 = 0;

            int[] dist = new int[points_number_octave0_model];
            int[,] dist2 = new int[points_number_octave0_model, 2];

            
            for (int i = 0; i < points_number_octave0_model; i++)
            {
               // dist[i] = diff_binary(point_input.sub_polar_gray.Data, points_models[i].sub_polar_gray.Data, r, angl * 2) ;
                dist[i] = diff_binary(point_input.binary_discriptor2, points_models[i].binary_discriptor2, n_bits*n_bits_moments);

            }

            

            Min(dist, out min_distance, out index);
            //   Min(dist, out min_distance, out index);
            //   Min(dist2, out min_distance2, out index2);

            //   dist2 = _tri(dist, 10);

            matched_point.dist = min_distance;
            matched_point.point_input = point_input.pt;
            matched_point.point_model = points_models[index].pt;
            matched_point.tab_dist = dist2; 
            matched_point.index_model = index;

            return matched_point;



        }


        public Matched_Point[] matching_with_Coeff()
        {
            Matched_Point[] matched_points = new Matched_Point[points_number_octave0_input];

            for (int i = 0; i < points_number_octave0_input; i++)
            {
                matched_points[i] = match_points_with_Coeff(point_Descriptor_Input[i], point_Descriptor_Model);

                matched_points[i].index_input = i;

            }

            return matched_points;

        }

       public void get_metched_points()
        {
            matched_points = new Matched_Point[points_number_octave0_input];

            matched_points = matching_with_Coeff();

           // matching_by_opencv();

        }

       public float[] tri(float[] tab)
       {
           float min = 0, maxi = 0;
           int indice = 0;
           float[] tab2 = new float[tab.Length];

           Max(tab, out maxi);

           for (int i = 0; i < tab.Length; i++)
           {
               Min(tab, out min, out indice);
               // label3.Text = "" + min;
               tab2[i] = min;
               tab[indice] = maxi + 1;

           }
           return tab2;

       }
       public int[] tri(int[] tab)
       {
           int min = 0, maxi = 0;
           int indice = 0;
           int[] tab2 = new int[tab.Length];

           Max(tab, out maxi);

           for (int i = 0; i < tab.Length; i++)
           {
               Min(tab, out min, out indice);
               // label3.Text = "" + min;
               tab2[i] = min;
               tab[indice] = maxi + 1;

           }
           return tab2;

       }


       public int[] tri(int[] tab, int max_index)
       {
           int min = 0, maxi = 0;
           int indice = 0;
           int[] tab2 = new int[tab.Length];

           Max(tab, out maxi);

           for (int i = 0; i < max_index; i++)
           {
               Min(tab, out min, out indice);
               // label3.Text = "" + min;
               tab2[i] = min;
               tab[indice] = maxi + 1;

           }
           return tab2;

       }


       public int[,] _tri(int[] tab, int max_index)
       {

           int min = 0, maxi = 0;
           int indice = 0;
           int[,] tab2 = new int[max_index, 2];

           Max(tab, out maxi);

           for (int i = 0; i < max_index; i++)
           {

               Min(tab, out min, out indice);

               tab2[i, 0] = min;
               tab2[i, 1] = indice;
               tab[indice] = maxi + 1;

           }
           return tab2;

       }

        public void Min(float[] tab, out float min, out int index)
        {
            min = tab[0];
            index = 0;
            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] < min)
                {
                    min = tab[i];
                    index = i;
                }
            }
        }
        public void Min(double[] tab, out double min, out int index)
        {
            min = tab[0];
            index = 0;
            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] < min)
                {
                    min = tab[i];
                    index = i;
                }
            }
        }

        public void Min(float[,] tab, int n, out float min, out int index)
        {
            min = tab[0, 1];
            index = 0;
            for (int i = 0; i < n; i++)
            {
                if (tab[i, 1] < min)
                {
                    min = tab[i, 1];
                    index = i;
                }
            }
        }

        public void Min(int[] tab, out int min, out int index)
        {
            min = tab[0];
            index = 0;
            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] < min)
                {
                    min = tab[i];
                    index = i;
                }
            }
        }

        public void Min(float[, ,] diffs, int lenght1, int lenght2, int lenght3, out float min, out int index1, out int index2)
        {
            min = 0;
            for (int p = 0; p < lenght3; p++)
                min += diffs[0, 0, p];

            index1 = 0;
            index2 = 0;

            float sum = 0;
            for (int i = 0; i < lenght1; i++)
                for (int j = 0; j < lenght2; j++)
                {
                    sum = 0;
                    for (int p = 0; p < lenght3; p++)
                    {
                        sum += diffs[i, j, p];

                    }

                    if (sum < min)
                    {
                        min = sum;
                        index1 = i;
                        index2 = j;

                    }

                }
        }

        public void Max(float[] tab, out float max)
        {
            max = tab[0];

            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] > max)
                {
                    max = tab[i];

                }
            }
        }

        public void Max(float[,] tab, int n, out float max)
        {
            max = tab[0, 1];

            for (int i = 0; i < n; i++)
            {
                if (tab[i, 1] > max)
                {
                    max = tab[i, 1];

                }
            }
        }

        public void Max(int[] tab, out int index)
        {
            int max = tab[0];
            index = 0;

            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] > max)
                {
                    max = tab[i];
                    index = i;
                }
            }
        }

        public void Max(int[] tab, out int max, out int index)
        {
            max = tab[0];
            index = 0;

            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] > max)
                {
                    max = tab[i];
                    index = i;
                }
            }
        }

        public void Max(float[, ,] diffs, int lenght1, int lenght2, int lenght3, out float max)
        {
            max = 0;
            for (int p = 0; p < lenght3; p++)
                max += diffs[0, 0, p];

            float sum = 0;

            for (int i = 0; i < lenght1; i++)
                for (int j = 0; j < lenght2; j++)
                {
                    sum = 0;

                    for (int p = 0; p < lenght3; p++)
                        sum += diffs[i, j, p];

                    if (sum > max)
                    {
                        max = sum;
                    }
                }




        }

        public void max(float[] tab, out float max, out int index)
        {
            max = tab[0];
            index = 0;
            for (int i = 0; i < tab.Length; i++)
            {
                if (tab[i] > max)
                {
                    max = tab[i];
                    index = i;
                }
            }
        }


        int[] histogram = new int[256];
        int max_hist = 1;
        int t = 0;

        private int calculateThreshold(Image<Gray, byte> frame_gray)
        {

            //0.2125
            //0.7154
            //0.0721

            calculateHistogramm(frame_gray);
           


            float[] vet = new float[256];
            //int[] hist = new int[256];
            vet.Initialize();

            float p1, p2, p12;
            int k;


            for (k = 1; k != 255; k++)
            {
                p1 = Px(0, k, histogram);
                p2 = Px(k + 1, 255, histogram);
                p12 = p1 * p2;
                if (p12 == 0)
                    p12 = 1;
                float diff = (Mx(0, k, histogram) * p2) - (Mx(k + 1, 255, histogram) * p1);
                vet[k] = (float)diff * diff / p12;
                //vet[k] = (float)Math.Pow((Mx(0, k, hist) * p2) - (Mx(k + 1, 255, hist) * p1), 2) / p12;
            }



           return t = findMax(vet, 256);

          
        }

        void calculateHistogramm(Image<Gray, byte> frame_gray)
        {

            histogram = new int[256];

            max_hist = 0;
            int val = 0;

            for (int i = 0; i < frame_gray.Width; i++)
                for (int j = 0; j < frame_gray.Height; j++)
                {


                    if ((histogram[val = frame_gray.Data[j, i, 0]]++) > max_hist)
                        max_hist = histogram[val];

                }

            
            for (int i = 0; i < 256; i++)
                if ((histogram[i]) > max_hist)
                    max_hist = histogram[i];
            
        }

        // function is used to compute the q values in the equation
        private float Px(int init, int end, int[] hist)
        {
            int sum = 0;
            int i;
            for (i = init; i <= end; i++)
                sum += hist[i];

            return (float)sum;
        }

        // function is used to compute the mean values in the equation (mu)
        private float Mx(int init, int end, int[] hist)
        {
            int sum = 0;
            int i;
            for (i = init; i <= end; i++)
                sum += i * hist[i];

            return (float)sum;
        }

        // finds the maximum element in a vector
        private int findMax(float[] vec, int n)
        {
            float maxVec = 0;
            int idx = 0;
            int i;

            for (i = 1; i < n - 1; i++)
            {
                if (vec[i] > maxVec)
                {
                    maxVec = vec[i];
                    idx = i;
                }
            }
            return idx;
        }


    }
}
