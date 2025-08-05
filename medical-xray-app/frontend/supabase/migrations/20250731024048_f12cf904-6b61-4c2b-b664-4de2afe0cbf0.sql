-- Create storage bucket for X-ray images
INSERT INTO storage.buckets (id, name, public) VALUES ('xray-images', 'xray-images', true);

-- Create table for X-ray analysis results
CREATE TABLE public.xray_analyses (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  original_image_url TEXT NOT NULL,
  processed_image_url TEXT,
  diagnosis TEXT NOT NULL,
  confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  details TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.xray_analyses ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (since this is a demo app)
CREATE POLICY "Anyone can view analyses" 
ON public.xray_analyses 
FOR SELECT 
USING (true);

CREATE POLICY "Anyone can create analyses" 
ON public.xray_analyses 
FOR INSERT 
WITH CHECK (true);

-- Create storage policies for X-ray images
CREATE POLICY "Anyone can view X-ray images" 
ON storage.objects 
FOR SELECT 
USING (bucket_id = 'xray-images');

CREATE POLICY "Anyone can upload X-ray images" 
ON storage.objects 
FOR INSERT 
WITH CHECK (bucket_id = 'xray-images');

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_xray_analyses_updated_at
BEFORE UPDATE ON public.xray_analyses
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();