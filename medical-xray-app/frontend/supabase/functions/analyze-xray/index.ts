import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const formData = await req.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      throw new Error('No file provided');
    }

    console.log('Processing X-ray image:', file.name, 'Size:', file.size);

    // Generate unique filename
    const timestamp = Date.now();
    const originalFileName = `original_${timestamp}_${file.name}`;
    const processedFileName = `processed_${timestamp}_${file.name}`;

    // Upload original image to storage
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('xray-images')
      .upload(originalFileName, file);

    if (uploadError) {
      console.error('Upload error:', uploadError);
      throw uploadError;
    }

    console.log('Original image uploaded:', uploadData.path);

    // Get public URL for original image
    const { data: { publicUrl: originalImageUrl } } = supabase.storage
      .from('xray-images')
      .getPublicUrl(originalFileName);

    // Simulate AI analysis processing
    await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay

    // Mock AI analysis results
    const diagnoses = [
      {
        diagnosis: "Potential pneumonia detected",
        confidence: 0.87,
        details: "A suspicious opacity was found in the lower lobe of the right lung. The shadowing pattern suggests possible pneumonia. Recommend clinical correlation and follow-up imaging."
      },
      {
        diagnosis: "Normal chest X-ray",
        confidence: 0.94,
        details: "No acute abnormalities detected. The lungs appear clear with normal cardiac silhouette and bone structures. Continue routine monitoring as clinically indicated."
      },
      {
        diagnosis: "Possible rib fracture",
        confidence: 0.72,
        details: "Linear lucency observed in the 6th rib on the left side, concerning for possible fracture. Recommend clinical correlation and consider additional imaging if symptomatic."
      },
      {
        diagnosis: "Enlarged heart shadow",
        confidence: 0.81,
        details: "Cardiomegaly noted with an enlarged cardiac silhouette. The cardiothoracic ratio appears increased. Recommend echocardiogram for further evaluation."
      }
    ];

    // Random selection for demo
    const randomResult = diagnoses[Math.floor(Math.random() * diagnoses.length)];

    // Create a simple processed image by copying the original
    // In a real scenario, this would be the AI-processed image with annotations
    const processedImageBlob = await file.arrayBuffer();
    const processedFile = new File([processedImageBlob], processedFileName, { type: file.type });

    // Upload processed image to storage
    const { data: processedUploadData, error: processedUploadError } = await supabase.storage
      .from('xray-images')
      .upload(processedFileName, processedFile);

    if (processedUploadError) {
      console.error('Processed upload error:', processedUploadError);
      throw processedUploadError;
    }

    // Get public URL for processed image
    const { data: { publicUrl: processedImageUrl } } = supabase.storage
      .from('xray-images')
      .getPublicUrl(processedFileName);

    // Save analysis to database
    const { data: analysisData, error: analysisError } = await supabase
      .from('xray_analyses')
      .insert({
        original_image_url: originalImageUrl,
        processed_image_url: processedImageUrl,
        diagnosis: randomResult.diagnosis,
        confidence: randomResult.confidence,
        details: randomResult.details
      })
      .select()
      .single();

    if (analysisError) {
      console.error('Analysis save error:', analysisError);
      throw analysisError;
    }

    console.log('Analysis completed and saved:', analysisData.id);

    return new Response(JSON.stringify({
      id: analysisData.id,
      diagnosis: randomResult.diagnosis,
      confidence: randomResult.confidence,
      details: randomResult.details,
      originalImageUrl,
      processedImageUrl,
      createdAt: analysisData.created_at
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in analyze-xray function:', error);
    return new Response(JSON.stringify({ 
      error: error.message || 'An error occurred during analysis' 
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});