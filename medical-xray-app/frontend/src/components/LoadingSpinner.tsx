import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Brain, Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = "Analyzing X-ray image..." 
}) => {
  return (
    <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
      <CardContent className="p-8">
        <div className="flex flex-col items-center gap-6 text-center">
          {/* Animated Brain Icon */}
          <div className="relative">
            <Brain className="h-16 w-16 text-primary animate-pulse" />
            <Loader2 className="absolute -top-1 -right-1 h-6 w-6 text-primary animate-spin" />
          </div>
          
          {/* Loading Message */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold">AI Analysis in Progress</h3>
            <p className="text-muted-foreground">{message}</p>
          </div>
          
          {/* Progress Indicator */}
          <div className="w-full max-w-sm">
            <div className="bg-muted rounded-full h-2">
              <div className="bg-primary h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
          </div>
          
          {/* Processing Steps */}
          <div className="text-xs text-muted-foreground space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span>Image uploaded successfully</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span>Pre-processing image data</span>
            </div>
            <div className="flex items-center gap-2">
              <Loader2 className="w-2 h-2 animate-spin text-primary" />
              <span>Running AI analysis model</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-muted-foreground rounded-full" />
              <span>Generating diagnostic report</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};