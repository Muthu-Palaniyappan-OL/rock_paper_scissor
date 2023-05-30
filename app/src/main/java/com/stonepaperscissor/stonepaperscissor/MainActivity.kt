package com.stonepaperscissor.stonepaperscissor

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.stonepaperscissor.stonepaperscissor.databinding.ActivityMainBinding
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.*


class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraController: LifecycleCameraController
    private lateinit var imageCapture: ImageCapture


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        findViewById<TextView>(R.id.textView).visibility = View.INVISIBLE

        findViewById<Button>(R.id.button).setOnClickListener(View.OnClickListener {
            cameraController.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    Log.d(TAG, "onCaptureSuccess: "+ image.planes[0].buffer.capacity())
                    Log.d(TAG, "onCaptureSuccess: "+ image.width)
                    Log.d(TAG, "onCaptureSuccess: "+ image.height)
                    val bitmap = convertImageProxyToBitmap(image)
                    bitmap?.let {
                        val scalledImage = Bitmap.createScaledBitmap(it, 300, 300, false);
                        classifyImage(scalledImage)
                    }
                    image.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    Toast.makeText(applicationContext, "Error: "+exception.message, Toast.LENGTH_LONG).show()
                }
            })
        })

        startCamera()
    }

    private fun startCamera() {
        val previewView: PreviewView = viewBinding.viewFinder
        cameraController = LifecycleCameraController(baseContext)
        cameraController.bindToLifecycle(this)
        cameraController.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        previewView.controller = cameraController
    }

    fun classifyImage(bitmap: Bitmap) :Unit {
        try {
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(resources.openRawResource(R.raw.rps).readBytes())
            val buffer = bitmapToFloatBuffer(bitmap)
            val inputTensor = OnnxTensor.createTensor(env, buffer, longArrayOf(1, 300, 300, 1))

            val result = session.run(mapOf("random_flip_2_input" to inputTensor))

            val map = mapOf<Int, String>(0 to "Rock \uD83E\uDEA8", 1 to "Paper \uD83D\uDCC4", 2 to "Scissor ✂\uFE0F")
            val textView = findViewById<TextView>(R.id.textView)
            textView.visibility = View.VISIBLE
            var max = 0
            val floatBuffer = (result[0] as OnnxTensor).floatBuffer
            val floatArray = FloatArray(floatBuffer.remaining())
            floatBuffer.get(floatArray)
            for(i in 0..2) {
                if (floatArray[max] < floatArray[i])
                    max = i
            }

            textView.text = map[max]

            textView.postDelayed(Runnable { textView.setVisibility(View.INVISIBLE) }, 1000)

        } catch (exception: Exception) {
            Log.d(TAG, "classifyImage: expection "+exception)
            Log.d(TAG, "classifyImage:  "+exception.message)
            Toast.makeText(applicationContext, "Error In Model Working: "+ exception.message, Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        private const val TAG = "CameraXApp"
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun convertImageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null

        val format = image.format
        return when (format) {
            ImageFormat.JPEG -> {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
            ImageFormat.YUV_420_888 -> {
                val yBuffer = image.planes[0].buffer
                val uBuffer = image.planes[1].buffer
                val vBuffer = image.planes[2].buffer
                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()
                val nv21 = ByteArray(ySize + uSize + vSize)

                yBuffer.get(nv21, 0, ySize)
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)

                val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                val outputStream = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, outputStream)
                val jpegBytes = outputStream.toByteArray()
                BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            }
            else -> null
        }
    }

    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        Log.d(TAG, "bitmapToByteBuffer:     "+bitmap.width +" " + bitmap.height)
        val byteBuffer = FloatBuffer.allocate((bitmap.width * bitmap.height))

        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                byteBuffer.put(((bitmap.getPixel(x, y) shr 16) and  0xFF) * 1f)
            }
        }

        byteBuffer.flip()

        return byteBuffer
    }

    fun printBuffer(buffer:ByteBuffer) {
        Log.d(TAG, "Buffer: ")
        while (buffer.hasRemaining()) {
            val value: Float = buffer.getFloat()
            Log.d(TAG, "printBuffer: " +(value))
        }
        buffer.rewind()
    }


}