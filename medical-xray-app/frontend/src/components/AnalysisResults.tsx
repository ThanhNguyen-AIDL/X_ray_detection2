import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Brain, Clock, TrendingUp } from 'lucide-react';

interface AnalysisResult {
  id: string;
  diagnosis: string;
  confidence: number;
  details: string;
  originalImageUrl: string;
  processedImageUrl: string;
  createdAt: string;
  // IOU metrics hidden for now
  // metrics?: {
  //   iou?: number;
  // };
}

interface AnalysisResultsProps {
  result: AnalysisResult;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result }) => {
  const getConfidenceColor = (confidence: number) => {
    // Now expecting values 0-100 directly
    if (confidence >= 80) return 'bg-green-500';
    if (confidence >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getConfidenceText = (confidence: number) => {
    // Now expecting values 0-100 directly
    if (confidence >= 80) return 'High';
    if (confidence >= 60) return 'Medium';
    return 'Low';
  };

  return (
    <div className="space-y-6">
      {/* Images Comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Image Analysis Comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground">Original Image</h3>
              <div className="relative rounded-lg overflow-hidden border-2 border-border">
                <img
                  src={result.originalImageUrl}
                  alt="Original X-ray"
                  className="w-full h-64 object-contain bg-muted"
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground">AI Processed Image</h3>
              <div className="relative rounded-lg overflow-hidden border-2 border-primary">
                <img
                  src={result.processedImageUrl}
                  alt="Processed X-ray"
                  className="w-full h-64 object-contain bg-muted"
                />
                <div className="absolute top-2 right-2">
                  <Badge variant="secondary" className="bg-primary text-primary-foreground">
                    AI Enhanced
                  </Badge>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Report */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            AI Analysis Report
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Diagnosis */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Diagnosis</h3>
            <p className="text-lg font-semibold">{result.diagnosis}</p>
          </div>

          <Separator />

          {/* Confidence Score */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Confidence Score</h3>
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${getConfidenceColor(result.confidence)}`}
                  style={{ width: `${result.confidence}%` }}
                />
              </div>
              <Badge variant="outline" className="font-mono">
                {(result.confidence).toFixed(1)}%
              </Badge>
              <Badge 
                variant={result.confidence >= 80 ? 'default' : result.confidence >= 60 ? 'secondary' : 'destructive'}
              >
                {getConfidenceText(result.confidence)}
              </Badge>
            </div>
          </div>

          <Separator />
          
          {/* IOU Score - Hidden for now 
          {result.metrics?.iou !== undefined && (
            <>
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground">
                  IOU Score (Intersection over Union)
                </h3>
                <div className="flex items-center gap-3">
                  <div className="flex-1 bg-muted rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${(result.metrics.iou * 100)}%` }}
                    />
                  </div>
                  <Badge variant="outline" className="font-mono">
                    {(result.metrics.iou * 100).toFixed(1)}%
                  </Badge>
                  <Badge variant="secondary">
                    Box Overlap Ratio
                  </Badge>
                </div>
              </div>
              <Separator />
            </>
          )}
          */}

          {/* Detailed Findings */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Detailed Findings</h3>
            <p className="text-sm leading-relaxed">{result.details}</p>
          </div>

          <Separator />

          {/* Analysis Metadata */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>Analysis completed: {new Date(result.createdAt).toLocaleString()}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};