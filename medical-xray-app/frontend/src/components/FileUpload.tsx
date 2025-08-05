import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileImage } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClearFile: () => void;
  disabled?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  selectedFile,
  onClearFile,
  disabled = false
}) => {
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image smaller than 10MB",
          variant: "destructive"
        });
        return;
      }

      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    },
    multiple: false,
    disabled,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  if (selectedFile) {
    return (
      <Card className="relative">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <FileImage className="h-5 w-5 text-primary" />
              <span className="text-sm font-medium truncate">{selectedFile.name}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearFile}
              disabled={disabled}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="relative rounded-lg overflow-hidden border-2 border-border">
            <img
              src={URL.createObjectURL(selectedFile)}
              alt="Selected X-ray"
              className="w-full h-48 object-contain bg-muted"
            />
          </div>
          
          <div className="mt-3 text-xs text-muted-foreground">
            Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`transition-all duration-200 ${
      isDragActive || dragActive 
        ? 'border-primary ring-2 ring-primary ring-opacity-20' 
        : 'border-dashed border-2'
    } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-primary'}`}>
      <CardContent className="p-8">
        <div {...getRootProps()} className="text-center">
          <input {...getInputProps()} />
          
          <div className="flex flex-col items-center gap-4">
            <div className={`p-4 rounded-full transition-colors ${
              isDragActive || dragActive 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-muted text-muted-foreground'
            }`}>
              <Upload className="h-8 w-8" />
            </div>
            
            <div className="space-y-2">
              <p className="text-lg font-medium">
                {isDragActive ? 'Drop the X-ray image here' : 'Upload X-ray Image'}
              </p>
              <p className="text-sm text-muted-foreground">
                Drag and drop or click to select
              </p>
              <p className="text-xs text-muted-foreground">
                Supports PNG, JPG, JPEG, BMP, TIFF (max 10MB)
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};