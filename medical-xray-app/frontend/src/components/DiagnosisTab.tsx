import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileUpload } from '@/components/FileUpload';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { AnalysisResults } from '@/components/AnalysisResults';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';
import { DetectionService } from '../integration';

interface AnalysisResult {
  id: string;
  diagnosis: string;
  confidence: number;
  details: string;
  originalImageUrl: string;
  processedImageUrl: string;
  createdAt: string;
  metrics?: {
    iou?: number;
  };
}

const DiagnosisTab = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setAnalysisResult(null); // Clear previous results
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    console.log('Starting analysis of file:', selectedFile.name);
    
    try {
      // Check API connectivity first
      try {
        const testResponse = await fetch('http://localhost:8000/', { 
          mode: 'cors',
          method: 'GET'
        });
        console.log('API connectivity test:', testResponse.ok ? 'Success' : 'Failed');
      } catch (connectionError) {
        console.error('API connectivity test error:', connectionError);
      }
      
      // Use our new DetectionService instead of Supabase
      console.log('Sending file to DetectionService...');
      const result = await DetectionService.detectSingleImage(selectedFile);
      
      // Map the API response to our expected format
      // Create full URLs for the images
      console.log('API returned image URL:', result.image_url);
      const apiBaseUrl = 'http://localhost:8000';
      
      // Convert relative URL to absolute URL for proper testing and display
      let imageUrl;
      if (result.image_url.startsWith('/')) {
        // Relative URL - prepend base URL
        imageUrl = `${apiBaseUrl}${result.image_url}`;
      } else {
        // Already absolute URL
        imageUrl = result.image_url;
      }
      
      // Test the image URL before using it
      try {
        const imgTest = await fetch(imageUrl, { method: 'HEAD', mode: 'cors' });
        if (!imgTest.ok) {
          console.warn(`Warning: Image URL returned ${imgTest.status}: ${imageUrl}`);
          console.log('Falling back to placeholder image');
          imageUrl = `${apiBaseUrl}/results/no_image_found.jpg`;
        } else {
          console.log('✅ Image URL test successful:', imageUrl);
        }
      } catch (imgError) {
        console.error('Error testing image URL:', imgError);
        console.log('Falling back to placeholder image');
        imageUrl = `${apiBaseUrl}/results/no_image_found.jpg`;
      }
      
      console.log('Final image URL to display:', imageUrl);
        
      const analysisData: AnalysisResult = {
        id: new Date().toISOString(),
        diagnosis: result.detections.length > 0 ? 
          result.detections.map(d => d.class_name).join(', ') : 
          'No conditions detected',
        confidence: result.detections.length > 0 ? 
          result.detections[0].confidence_score || result.detections[0].confidence * 100 : 0,
        details: `Found ${result.detection_count} conditions in ${result.processing_time.toFixed(2)}s`,
        originalImageUrl: selectedFile ? URL.createObjectURL(selectedFile) : '',
        processedImageUrl: imageUrl,
        createdAt: new Date().toISOString(),
        // metrics field is removed for now
        metrics: {
          iou: result.detections.length > 0 ? result.detections[0].metrics?.iou || 0 : 0
        }
      };

      setAnalysisResult(analysisData);
      
      toast({
        title: "Phân tích hoàn tất",
        description: "Hình ảnh X-quang đã được phân tích thành công bởi hệ thống AI của TMA.",
      });

    } catch (error) {
      console.error('Analysis error:', error);
      
      // Check for specific error types
      let errorMessage = "Có lỗi xảy ra khi phân tích X-quang.";
      if (error instanceof Error) {
        if (error.message.includes('NetworkError') || 
            error.message.includes('Failed to fetch')) {
          errorMessage = "Không thể kết nối đến API. Đảm bảo API server đang chạy (http://localhost:8000).";
        } else if (error.message.includes('CORS')) {
          errorMessage = "Lỗi CORS: API server không cho phép yêu cầu từ frontend.";
        }
      }
      
      toast({
        title: "Phân tích thất bại",
        description: errorMessage,
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>Tải lên hình ảnh X-quang</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClearFile={handleClearFile}
            disabled={isAnalyzing}
          />
          
          {selectedFile && (
            <div className="flex justify-center">
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                size="lg"
                className="w-full max-w-sm"
              >
                {isAnalyzing ? 'Đang phân tích...' : 'Phân tích hình ảnh'}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Loading State */}
      {isAnalyzing && (
        <LoadingSpinner message="AI của TMA đang phân tích kỹ lưỡng hình ảnh X-quang của bạn..." />
      )}

      {/* Results Section */}
      {analysisResult && !isAnalyzing && (
        <AnalysisResults result={analysisResult} />
      )}
    </div>
  );
};

export default DiagnosisTab;