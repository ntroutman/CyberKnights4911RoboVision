package org.kingsschools.cyberknights4911.steamworks.vision;

import org.opencv.core.Core;

public class GearVision {
	static {
		// Ensure the native library gets loaded
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

}
