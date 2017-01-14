package org.kingsschools.cyberknights4911.steamworks.vision;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.kingsschools.cyberknights4911.opencv.Imshow;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Main {
	private static String R_LEFT_L = "/retro-samples/sample-rotate/left_large.jpg";
	private static String R_LEFT_M = "/retro-samples/sample-rotate/left_med.jpg";
	private static String R_LEFT_S = "/retro-samples/sample-rotate/left_small.jpg";
	private static String R_STRAIGHT = "/retro-samples/sample-rotate/straight.jpg";
	private static String R_RIGHT_S = "/retro-samples/sample-rotate/right_small.jpg";
	private static String R_RIGHT_M = "/retro-samples/sample-rotate/right_med.jpg";
	private static String R_RIGHT_L = "/retro-samples/sample-rotate/right_large.jpg";

	public static void main(String[] args) {
		GearVision vision = new GearVision();

		List<Mat> rawImages = Arrays.asList(R_LEFT_L, R_LEFT_M, R_LEFT_S, R_STRAIGHT, R_RIGHT_S, R_RIGHT_M, R_RIGHT_L)
				.stream().map(path -> Highgui.imread(Main.class.getResource(path).getPath()))
				.collect(Collectors.toList());

		timeContours(vision, rawImages);
		timeContoursFast(vision, rawImages);
	}

	private static void timeContours(GearVision vision, List<Mat> rawImages) {
		double max = 0;
		long totalEllapsed = 0;
		final int count = 100;
		for (int i = 0; i < count; i++) {
			long start = System.nanoTime();
			for (Mat rawImage : rawImages) {
				Mat image = new Mat();
				Imgproc.resize(rawImage, image, new Size(640, 480));

				List<MatOfPoint> contours = vision.detectMarkers(image);

				for (MatOfPoint c : contours) {
					max = Imgproc.contourArea(c);
				}
			}
			long stop = System.nanoTime();
			System.out.print(".");
			totalEllapsed += (stop - start);
		}
		double avg = (totalEllapsed / count) / (1000_000d);
		System.out.println(max);
		System.out.println("Naive: Avg (ms): " + avg + ", FPS: " + 1 / (avg / 1000d));

	}

	private static void timeContoursFast(GearVision vision, List<Mat> rawImages) {
		double max = 0;
		long totalEllapsed = 0;
		final int count = 100;
		for (int i = 0; i < count; i++) {
			long start = System.nanoTime();
			Mat image = new Mat(); // use same target matrix for resized image
			for (Mat rawImage : rawImages) {

				Imgproc.resize(rawImage, image, new Size(640, 480), 0, 0, Imgproc.INTER_NEAREST);				
				List<MatOfPoint> contours = vision.detectMarkersFast(image);

				for (MatOfPoint c : contours) {
					max = Imgproc.contourArea(c);
				}
			}
			long stop = System.nanoTime();
			System.out.print(".");
			totalEllapsed += (stop - start);
		}
		double avg = (totalEllapsed / count) / (1000_000d);
		System.out.println(max);
		System.out.println("Fast: Avg (ms): " + avg + ", FPS: " + 1 / (avg / 1000d));

	}

	/**
	 * Simply load up some of the opencv packages and verify that things are
	 * installed correctly.
	 */
	public static void openCVInstallCheck() {
		System.out.println("Welcome to OpenCV " + Core.VERSION);
		Mat m = new Mat(5, 10, CvType.CV_8UC1, new Scalar(0));
		System.out.println("OpenCV Mat: " + m);
		Mat mr1 = m.row(1);
		mr1.setTo(new Scalar(1));
		Mat mc5 = m.col(5);
		mc5.setTo(new Scalar(5));
		System.out.println("OpenCV Mat data:\n" + m.dump());
	}
}
