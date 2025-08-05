// API integration with the FastAPI backend
const API_URL = 'http://localhost:8000';

export const DetectionService = {
  /**
   * Detect conditions in a single X-ray image
   * @param {File} imageFile - The X-ray image file
   * @returns {Promise} - Detection results
   */
  async detectSingleImage(imageFile) {
    try {
      console.log('Sending image to API:', imageFile.name);
      
      const formData = new FormData();
      formData.append('file', imageFile);
      
      // Use more verbose fetch options with proper CORS settings
      const response = await fetch(`${API_URL}/detect/single`, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        credentials: 'omit', // Don't send credentials for now
        headers: {
          'Accept': 'application/json',
        }
      });
      
      console.log('API response status:', response.status);
      
      if (!response.ok) {
        console.error('API error response:', response.statusText);
        throw new Error(`Detection failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('API response data:', data);
      return data;
    } catch (error) {
      console.error('Error detecting image:', error);
      throw error;
    }
  },
  
  /**
   * Start batch processing of images
   * @param {string} folderPath - Path to folder containing X-ray images
   * @returns {Promise} - Job information
   */
  async startBatchProcessing(folderPath) {
    try {
      const formData = new FormData();
      formData.append('folder_path', folderPath);
      
      const response = await fetch(`${API_URL}/detect/batch`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Batch processing failed: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error starting batch process:', error);
      throw error;
    }
  },
  
  /**
   * Check status of a batch job
   * @param {string} jobId - ID of the batch job
   * @returns {Promise} - Job status and results
   */
  async checkJobStatus(jobId) {
    try {
      const response = await fetch(`${API_URL}/jobs/${jobId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to get job status: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error checking job status:', error);
      throw error;
    }
  }
};

export default DetectionService;
