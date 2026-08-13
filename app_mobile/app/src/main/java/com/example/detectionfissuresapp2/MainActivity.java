package com.example.detectionfissuresapp2;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "CrackDetection";
    private static final String MODEL_FILE = "mobilenetv2_fissures.tflite";
    private static final String MODEL_MATERIAUX = "modele_concrete.tflite";    private static final int INPUT_SIZE = 224;
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};

    // Vues principales
    private PreviewView previewView;
    private ImageView capturedImage;
    private TextView resultText;
    private Button captureButton;
    private Button newPhotoButton;

    // Vues matériaux
    private Button analyserMateriaux;
    private ScrollView resultPanel;
    private EditText ageBatiment, humidite, temperature, usure, nbFissures;
    private TextView resultatMateriaux;

    // Modèles
    private Interpreter tflite;
    private Interpreter tfliteMateriaux;
    private ExecutorService cameraExecutor;
    private ImageCapture imageCapture;
    private Bitmap currentBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialiser les vues principales
        previewView = findViewById(R.id.previewView);
        capturedImage = findViewById(R.id.capturedImage);
        resultText = findViewById(R.id.resultText);
        captureButton = findViewById(R.id.captureButton);
        newPhotoButton = findViewById(R.id.newPhotoButton);

        // Initialiser les vues matériaux
        analyserMateriaux = findViewById(R.id.analyserMateriaux);
        resultPanel = findViewById(R.id.resultPanel);
        ageBatiment = findViewById(R.id.ageBatiment);
        humidite = findViewById(R.id.humidite);
        temperature = findViewById(R.id.temperature);
        usure = findViewById(R.id.usure);
        nbFissures = findViewById(R.id.nbFissures);
        resultatMateriaux = findViewById(R.id.resultatMateriaux);

        // État initial
        captureButton.setVisibility(View.VISIBLE);
        newPhotoButton.setVisibility(View.GONE);
        previewView.setVisibility(View.VISIBLE);
        capturedImage.setVisibility(View.GONE);
        resultText.setVisibility(View.GONE);
        resultPanel.setVisibility(View.GONE);

        // Configurer les boutons
        captureButton.setOnClickListener(v -> captureImage());
        newPhotoButton.setOnClickListener(v -> resetToCamera());
        analyserMateriaux.setOnClickListener(v -> analyserMateriaux());

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Vérifier les permissions
        if (allPermissionsGranted()) {
            initTensorFlowLite();
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initTensorFlowLite();
                startCamera();
            } else {
                Toast.makeText(this, "Permission camera refusee", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void initTensorFlowLite() {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setUseXNNPACK(false);
            options.setNumThreads(1);

            tflite = new Interpreter(loadModelFile(MODEL_FILE), options);
            tfliteMateriaux = new Interpreter(loadModelFile(MODEL_MATERIAUX), options);

            Log.d(TAG, "Modeles charges");
            runOnUiThread(() -> Toast.makeText(this, "Modeles charges", Toast.LENGTH_SHORT).show());

        } catch (Exception e) {
            Log.e(TAG, "Erreur chargement modeles", e);
            runOnUiThread(() -> Toast.makeText(this, "Erreur: " + e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    private MappedByteBuffer loadModelFile(String filename) throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd(filename);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder().build();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                Log.d(TAG, "Camera demarree");

            } catch (Exception e) {
                Log.e(TAG, "Erreur demarrage camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {
        if (imageCapture == null) return;

        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                currentBitmap = imageProxyToBitmap(image);

                runOnUiThread(() -> {
                    capturedImage.setImageBitmap(currentBitmap);
                    capturedImage.setVisibility(View.VISIBLE);
                    previewView.setVisibility(View.GONE);
                    captureButton.setVisibility(View.GONE);
                    newPhotoButton.setVisibility(View.VISIBLE);
                    resultText.setVisibility(View.VISIBLE);
                    resultPanel.setVisibility(View.VISIBLE);
                });

                detectCracks(currentBitmap);
                image.close();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Erreur capture", exception);
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Erreur de capture", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    private void resetToCamera() {
        runOnUiThread(() -> {
            previewView.setVisibility(View.VISIBLE);
            resultPanel.setVisibility(View.GONE);
            captureButton.setVisibility(View.VISIBLE);
            newPhotoButton.setVisibility(View.GONE);
            resultText.setText("");
            resultatMateriaux.setText("");
            ageBatiment.setText("");
            humidite.setText("");
            temperature.setText("");
            usure.setText("");
            nbFissures.setText("");
            currentBitmap = null;
        });
    }

    private void detectCracks(Bitmap bitmap) {
        if (tflite == null) return;

        try {
            ByteBuffer input = preprocessImage(bitmap);
            float[][] output = new float[1][1];
            tflite.run(input, output);

            float probability = output[0][0];
            float SEUIL = 0.57f;

            String result;
            if (probability > SEUIL) {
                if (probability > 0.85f) {
                    result = String.format(Locale.FRENCH,
                            "FISSURE GRAVE !\nURGENT : Inspection immediate\nConfiance: %.1f%%", probability * 100);
                } else if (probability > 0.70f) {
                    result = String.format(Locale.FRENCH,
                            "FISSURE MODEREE\nSurveillance recommandee\nConfiance: %.1f%%", probability * 100);
                } else {
                    result = String.format(Locale.FRENCH,
                            "FISSURE LEGERE\nA surveiller\nConfiance: %.1f%%", probability * 100);
                }
            } else {
                result = String.format(Locale.FRENCH,
                        "SAIN\nAucune fissure detectee\nConfiance: %.1f%%", (1 - probability) * 100);
            }

            runOnUiThread(() -> resultText.setText(result));

        } catch (Exception e) {
            Log.e(TAG, "Erreur detection", e);
        }
    }

    private void analyserMateriaux() {
        if (tfliteMateriaux == null) {
            runOnUiThread(() -> resultatMateriaux.setText("Modele materiaux non charge"));
            return;
        }

        try {
            // Récupérer les valeurs saisies
            float age = Float.parseFloat(ageBatiment.getText().toString());
            float hum = Float.parseFloat(humidite.getText().toString());
            float temp = Float.parseFloat(temperature.getText().toString());
            float u = Float.parseFloat(usure.getText().toString());
            float nbF = Float.parseFloat(nbFissures.getText().toString());

            // 🔴 PRÉPARER L'ENTRÉE (8 features)
            float[][] inputRaw = new float[1][8];

            // Valeurs brutes (à normaliser)
            inputRaw[0][0] = 300f;      // Ciment (kg)
            inputRaw[0][1] = 0f;        // Laitier (kg)
            inputRaw[0][2] = 0f;        // Cendres volantes (kg)
            inputRaw[0][3] = 180f;      // Eau (kg)
            inputRaw[0][4] = 6f;        // Superplastifiant (kg)
            inputRaw[0][5] = 970f;      // Granulat gros (kg)
            inputRaw[0][6] = 770f;      // Granulat fin (kg)
            inputRaw[0][7] = age;       // Age (jours)
            // 🔴 NORMALISATION (valeurs moyennes et écarts-types du dataset UCI Concrete)
            float[] mean = {281.17f, 73.90f, 54.19f, 181.57f, 6.20f, 972.92f, 773.58f, 45.66f};
            float[] std = {104.51f, 86.28f, 64.00f, 21.36f, 5.97f, 77.75f, 80.18f, 63.17f};

            float[][] input = new float[1][8];
            for (int i = 0; i < 8; i++) {
                input[0][i] = (inputRaw[0][i] - mean[i]) / std[i];
            }

            float[][] output = new float[1][1];
            tfliteMateriaux.run(input, output);
            float resistance = output[0][0];

            String resultat;
            if (resistance < 20) {
                resultat = "RESISTANCE TRES FAIBLE : " + String.format("%.1f", resistance) + " MPa\nRisque structurel eleve !";
            } else if (resistance < 30) {
                resultat = "RESISTANCE MODEREE : " + String.format("%.1f", resistance) + " MPa\nSurveillance recommandee";
            } else if (resistance < 50) {
                resultat = "BONNE RESISTANCE : " + String.format("%.1f", resistance) + " MPa\nStructure saine";
            } else {
                resultat = "RESISTANCE EXCELLENTE : " + String.format("%.1f", resistance) + " MPa\nStructure de haute qualite";
            }

            runOnUiThread(() -> resultatMateriaux.setText(resultat));

        } catch (Exception e) {
            Log.e(TAG, "Erreur analyse materiaux", e);
            runOnUiThread(() -> resultatMateriaux.setText("Erreur: " + e.getMessage()));
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        ByteBuffer input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            input.putFloat(r);
            input.putFloat(g);
            input.putFloat(b);
        }

        return input;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) tflite.close();
        if (tfliteMateriaux != null) tfliteMateriaux.close();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}